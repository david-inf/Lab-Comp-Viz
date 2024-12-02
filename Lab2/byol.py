import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import MLP


def loss_fn(x, y):
    # x (view1), y (view2): [N, 512]
    # la norma sulla seconda (ultima) dimensione è 1 norm([x_1,...,x_512])=1
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    # loss = (x - y)^2 = x^2 + y^2 - 2xy = 1 + 1 - 2xy = 2(1 - xy)
    # questa loss è element-wise per il batch raddoppiato
    # va a valutare soltanto le positive pair
    return 2 * (1 - (x * y).sum(dim=-1))  # exit with [N]


def update_moving_average(online_net, target_net, beta):
    # target_net: target network params
    # online_net: online network params
    # beta: decay

    def update_average(old, new, beta):
        # old: old params (target net) to consider
        # new: new params (online net) to consider

        # momentum update, chiamato per ogni parametro della rete target
        # aggiorna i parametri sulla base della combinazione convessa
        if old is None:
            return new
        return beta * old + (1 - beta) * new

    nets_params = zip(online_net.parameters(), target_net.parameters())
    for online_net_params, target_net_params in nets_params:
        # get target and online nets params
        old_weight, up_weight = online_net_params.data, target_net_params.data
        # update target net params (update class attribute)
        target_net_params.data = update_average(old_weight, up_weight, beta)


class BYOL(nn.Module):
    def __init__(self, backbone, beta=0.99):
        super().__init__()
        # self.device = device
        self.ma_decay = beta  # EMA decay

        ## Online network
        self.online_net = backbone  # ResNet18
        # metto un identità così da avere accesso al latent space di dim=512
        self.online_net.fc = nn.Identity()
        # actually, the predictor
        self.online_projector = MLP(512, 512, 4096)
        # self.online_net = self.online_net.to(self.device)
        # self.online_projector = self.online_projector.to(self.device)

        ## Target network
        self.target_net = None  # così viene istanziata uguale alla online

    def _get_target_encoder(self):
        # create target net if is None
        if self.target_net is None:
            # copy online net
            target_net = copy.deepcopy(self.online_net)
            # freeze params, update is through ema
            for p in target_net.parameters():
                p.requires_grad = False
            self.target_net = target_net
            # self.target_net = self.target_net.to(self.device)
        else:
            target_net = self.target_net
        return target_net

    def update_moving_average(self):
        # chiama la funzione definita sopra
        # l'aggiornamento della target viene fatto coi parametri dell'encoder
        # della online, proprio perché sono uguali in numero
        update_moving_average(self.online_net, self.target_net, self.ma_decay)

    def forward(self, x1, x2):
        # x1, x2: viste con augmentation

        images = torch.cat((x1, x2), dim=0)  # [2N, 3, 32, 32]

        # Encoder + Projector sulle due viste concatenate
        online_projections = self.online_projector(self.online_net(images))  # [2N, 512]
        # separa le proiezioni delle due view in [N, 512] e [N, 512]
        online_pred_one, online_pred_two = online_projections.chunk(2, dim=0)

        with torch.no_grad():
            # get target network
            target_net = self._get_target_encoder()  # prende la target che non si aggiorna
            # encoding in [2N, 512] e toglie dal grafo computazionale
            target_projections = target_net(images).detach()
            # separa per avere [N, 512] e [N, 512]
            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)

        # win-win having doubled the samples for each mini-batch
        # loss([N, 512], [N, 512]) viene fatta a coppie
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())  # for view1 [N]
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())  # for view2 [N]
        loss = loss_one + loss_two  # [N]  non serve /2??
        # il forward finisce col calcolo della loss

        # questo funziona se N è uguale per entrambe le viste, che di fatti è così
        # return loss_one, loss_two  # lungo la dimensione con N
        return loss.mean()
