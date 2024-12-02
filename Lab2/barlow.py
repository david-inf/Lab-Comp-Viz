import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        # self.device = device
        self.lambd = lambd  # for the redundancy reduction term

        # barlow twins projector (espande la rappresentazione)
        sizes = [512, 2048] # in the paper is [2048, 8192, 8192, 8192]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # TODO: find a better way to put on device
        # magari possono mettere in questo file la scelta del device direttamente
        # self.projector = self.projector.to(self.device)

        # normalization layer for the representations z1 and z2
        # evita sbilanciamento dei vettori all'interno dei batch
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        # self.bn = self.bn.to(self.device)

    def forward(self, features1, features2):

        x1 = self.bn(self.projector(features1))  # z^A=[N, 512]
        x2 = self.bn(self.projector(features2))  # z^B=[N, 512]
        x1 = F.normalize(x1, dim=1)  # z^A=[N, 512]
        x2 = F.normalize(x2, dim=1)  # z^B=[N, 512]

        # cross-correlation matrix (similarity matrix between features)
        # norm_i = torch.sqrt(torch.sum(features1**2, dim=0))  # [512]
        # norm_j = torch.sqrt(torch.sum(features2**2, dim=0))  # [512]
        cross_corr_matrix = x1.matmul(x2.T)  # C=[N, N]
        # print(cross_corr_matrix)

        invariance = torch.sum((1 - torch.diagonal(cross_corr_matrix)).pow_(2))
        # for i in range(cross_corr_matrix.shape[0]):
        #     C_ii = cross_corr_matrix[i, i]
        #     invariance += (1 - C_ii)**2

        # redundancy_reduction = torch.sum((cross_corr_matrix - torch.diag(torch.diagonal(cross_corr_matrix)))**2)
        redundancy_reduction = torch.sum(off_diagonal(cross_corr_matrix).pow_(2))
        # for i in range(cross_corr_matrix.shape[0]):
        #     for j in range(cross_corr_matrix.shape[1]):

        loss = invariance + self.lambd * redundancy_reduction
        return loss
