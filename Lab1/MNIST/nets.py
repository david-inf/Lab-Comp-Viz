# -*- coding: utf-8 -*-
"""
Neural Networks

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, net="Net1"):
        """ from torch.docs
            torch.nn.Linear(in_features, out_features, bias=True, device=None,
                            dtype=None)
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                            padding=0, dilation=1, groups=1, bias=True,
                            padding_mode='zeros', device=None, dtype=None)
            torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
                               return_indices=False, ceil_mode=False)
        """
        super(Net, self).__init__()
        self.net = net

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

        self.fc1_net2 = nn.Linear(1600, 128)

        self.conv1_net3 = nn.Conv2d(1, 16, 3, 1, 2)
        self.conv2_net3 = nn.Conv2d(16, 32, 3, 1)
        self.fc1_net3 = nn.Linear(2304, 128)


    def forward(self, x):
        # x.shape B, N, H, W

        if self.net == "Net1":
            ## Convolution layers
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            ## Network layers
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            # last x has the logits

        elif self.net == "Net2":
            ## Convolution layers
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            ## Network layers
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1_net2(x))
            x = self.fc2(x)

        elif self.net == "Net3":
            ## Convolution layers
            x = self.relu(self.conv1_net3(x))
            x = self.relu(self.conv2_net3(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            ## Network layers
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1_net3(x))
            x = self.fc2(x)

        # la log(softmax) serve poi in fase di calcolo della CE loss
        output = F.log_softmax(x, dim=1)

        return output
