import copy

import torch
import torch.nn as nn
from torchinfo import summary

from rich.console import Console
console = Console()


def visualize(model, input_data):
    out = model(input_data)
    console.print(f'Computed output, shape={out.shape}')
    model_stats = summary(
        model, input_data=input_data,
        col_names=[
            "input_size", "output_size", "num_params",
            # "params_percent", "kernel_size", "mult_adds",
            ],
        row_settings=("var_names",), col_width=18, depth=8, verbose=0)
    console.print(model_stats)


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


class BTNet(nn.Module):
    """ Barlow Twins final model """
    def __init__(self, backbone, projector, classifier):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)  # latent space
        x = self.projector(x)  # representation space
        x = self.classifier(x)  # classification head
        return x


class Block(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            # prima convoluzione
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_filters,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            # seconda convoluzione in cui la dimensione del tensore resta invariata
            # quello che cambia (immagino) sono i valori dei pixel della feature map
            # nn.Conv2d(in_channels=num_filters,
            #           out_channels=num_filters,
            #           kernel_size=kernel_size,
            #           padding='same'),
            # nn.BatchNorm2d(num_filters),
            # nn.ReLU(),
            # dimezza la feature map per il blocco successivo
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleNet(nn.Module):
    def __init__(self, num_filters, mlp_size, num_classes):
        super().__init__()
        # convolutional blocks
        self.blocks = nn.Sequential(
            Block(3, num_filters, (3,3)),
            Block(num_filters, num_filters*2, (3,3)),
            # Block(num_filters*2, num_filters*4, (3,3)),
            # Block(num_filters*4, num_filters*8, (3,3)),
        )
        # reduce convolution filters
        self.bottleneck = nn.Conv2d(
            in_channels=num_filters*2,
            out_channels=num_filters,
            kernel_size=(1,1)
        )
        # flatten convolution output
        self.flatten = nn.Flatten()  # Unroll the last feature map into a vector
        # mlp head
        self.mlp = nn.Sequential(
            nn.Linear(in_features=8*8*num_filters, out_features=2*mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=2*mlp_size, out_features=mlp_size),
            nn.ReLU()
        )
        self.fc = nn.Linear(in_features=mlp_size, out_features=num_classes)

    def forward(self, x):
        h = x
        # convolution
        h = self.blocks(h)
        # reduce channels
        h = self.bottleneck(h)
        # flatten
        # print(h.shape)
        h = self.flatten(h)
        # go to classification head
        h = self.mlp(h)
        h = self.fc(h)  # logits
        return h
