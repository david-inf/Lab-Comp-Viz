import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features):
        # features: torch.cat((x1,x2), dim=0)  # [2N, 512]

        # normalize features to later compute cosine distance/similarity btw them
        features = F.normalize(features, dim=1)  # ||f(x)||=1 => sim(\eta_{ij})=z_i^Tz_j

        ### Compute the similarity matrix btw features -> [2N x 2N]
        batch_size = features.shape[0] // 2
        similarity_matrix = features.matmul(features.T)  # [2N, 2N]
        # Si utilizzano delle maschere per eliminare parti della matrice
        # utilizzare una maschera per eliminare la diagonale principale

        ### Create the logits tensor
        #   - in the first position there is the similarity of the positive pair
        #   - in the other 2N-1 (mi sa 2N-2) positions there are the similarity w negatives
        # the shape of the tensor needs to be 2Nx2N-1, with N is the batch size
        # ciascun esempio dei 2N ha 2N-1 similarità (sulle colonne)
        logits = torch.zeros(2 * batch_size, 2 * batch_size - 1)
        # ********* #
        ## Not working
        # for i in range(size):  # take all samples
        #     sim_row = similarity_matrix[i, :]
        #     logits_row = torch.cat([sim_row[:i], sim_row[i+1:]])
        #     logits[i, :] = logits_row
        # ********* #
        start = time.time()
        for idx, val in enumerate(similarity_matrix):
            # idx: indice della riga
            # val: riga della similarity_matrix
            row = torch.zeros(2 * batch_size - 1)
            # positive pair index
            pos_idx = idx + batch_size if idx < batch_size else idx - batch_size

            row[0] = val[pos_idx]  # positive similarity nella prima posizione
            row[1:] = torch.tensor([v for i, v in enumerate(val) if i != idx and i != pos_idx])

            logits[idx] = row
        # ********* #

        # to compute the contrastive loss using the CE loss, we just need to
        # specify where is the similarity of the positive pair in the logits tensor
        # since we put in the first position we create a gt of all zeros
        # N.B.: this is just one of the possible implementations!
        gt = torch.zeros(logits.shape[0], dtype=torch.long)
        ### Loss matrix l_{ij}
        # loss_matrix = torch.zeros(size, size)
        # for i in range(size):
        #     for j in range(size):
        #         loss_ij = self.criterion()

        loss = self.criterion(logits, gt)

        return loss
