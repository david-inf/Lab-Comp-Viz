import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def normalize_repr(z):
    # z: [N, M]
    # standardize across batch dimension
    mean = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True)
    return (z - mean) / std


class BarlowTwins(nn.Module):
    def __init__(self, lambd, z_dim=512, sizes=[1024]):
        super().__init__()
        self.lambd = lambd  # for the redundancy reduction term

        # barlow twins projector (espande la rappresentazione)
        sizes = [z_dim] + sizes # in the paper is [2048, 8192, 8192, 8192]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        # evita sbilanciamento dei vettori all'interno dei batch
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, features1, features2):

        x1 = self.bn(self.projector(features1))  # z^A=[N, 512]
        x2 = self.bn(self.projector(features2))  # z^B=[N, 512]
        # x1 = F.normalize(x1, dim=0)  # z^A=[N, 512] normalize along feature dim
        # x2 = F.normalize(x2, dim=0)  # z^B=[N, 512] the paper says along batch dim
        x1 = normalize_repr(x1)
        x2 = normalize_repr(x2)

        # cross-correlation matrix (similarity matrix between features)
        size = x1.shape[0]
        cross_corr_matrix = x1.T @ x2  # C=[512, 512]
        cross_corr_matrix.div_(size)

        invariance = torch.diagonal(cross_corr_matrix).add_(-1).pow_(2).sum()
        redundancy_reduction = off_diagonal(cross_corr_matrix).pow_(2).sum()
        loss = invariance + self.lambd * redundancy_reduction

        return loss
