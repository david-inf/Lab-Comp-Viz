import copy

import torch
import torch.nn as nn


def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    # MLP class for projector and predictor
    return nn.Sequential(
        nn.Linear(dim, hidden_size),  # latent dim -> hidden size
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)  # hidden size -> latent dim
    )


class SiameseNetSync(nn.Module):
    def __init__(self, backbone):
        # passare da fuori la backbone almeno è possibile utilizzarlo
        # ovvero una volta addestrata la backbone si può poi utilizzare
        super().__init__()
        self.encoder = backbone
        self.in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

    def forward(self, x1, x2, return_dict=True):
        # x1, x2: augmentation 1 and 2
        batch_size = x1.shape[0]

        views = torch.concat([x1, x2], dim=0)
        f = self.encoder(views)  # [2N, 3, ...]

        x1, x2 = torch.split(f, batch_size, dim=0)

        if return_dict:
            return {'view1': x1, 'view2': x2}

        return f


class SiameseNetAsync(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder1 = backbone
        self.encoder1.fc = nn.Identity()

        # start with same params, but the update may be different
        self.encoder2 = copy.deepcopy(backbone)
        self.encoder2.fc = nn.Identity()

    def forward(self, x1, x2, return_dict=True):
        # x1, x2: augmentation 1 and 2
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)

        if return_dict:
            return {'view1': x1, 'view2': x2}

        return x1, x2
