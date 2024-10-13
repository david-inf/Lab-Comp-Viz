# -*- coding: utf-8 -*-
"""


"""

import yaml
import argparse
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

from nets import Net
from utils import train_loop, diagnostic


# %% Get data

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
      ])

dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('./data', train=False, transform=transform)


# %% Train the model

def main(opts):
    # Run a single model with options from a yaml file

    device = torch.device(opts.device) # to use the GPU

    ## Loaders
    train_loader = torch.utils.data.DataLoader(
        dataset1, num_workers=0, pin_memory=True, shuffle=True,
        batch_size=opts.batch_size)
    test_loader = torch.utils.data.DataLoader(
        dataset2, num_workers=0, pin_memory=True, shuffle=True,
        batch_size=opts.batch_size)

    ## Training
    model = Net(opts.model_name)
    print(model.net)

    losses_train, accs_train, losses_test, accs_test = train_loop(
        train_loader, test_loader, model, device, opts.lr, opts.momentum,
        opts.nesterov, opts.max_epochs)

    ## Diagnostic
    diagnostic(opts.max_epochs, losses_train, accs_train, losses_test, accs_test)
    plt.savefig(opts.save_to)
    print(f"Saved to {opts.save_to}")


if __name__ == "__main__":

    ### If options are passed through a yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML Configuration file")

    # store parsed arguments into args for clarity
    args = parser.parse_args()
    with open(args.config, "r") as f:
        # ensures the file il properly closed afeter reading
        # SafeLoader is generally preferred
        opts = yaml.load(f, Loader=yaml.SafeLoader)

    ### If options are passed through a dictionary, use for debugging
    # opts = dict(lr=0.01, momentum=0., nesterov=False, batch_size=64, max_epochs=5)
    # opts = dict(lr=[0.01, 0.001], momentum=[0., 0.9], nesterov=False,
    #             batch_size=64, max_epochs=5)

    opts = SimpleNamespace(**opts)

    # with launch_ipdb_on_exception():
    main(opts)
