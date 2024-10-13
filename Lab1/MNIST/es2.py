# -*- coding: utf-8 -*-
"""
Es 2, work on optimization algorithm

"""

import yaml
import argparse
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

from nets import Net
from utils import train_loop, diagnostic, multiple_diagnostic


# %% Get data

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
      ])

dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('./data', train=False, transform=transform)


# %% Train the model

def main(solvers_opts):
    # Run multiple models with different lr and momentum

    # batch_size = 64
    max_epochs = 20

    device = torch.device("cuda")

    ## Loaders
    # train_loader = torch.utils.data.DataLoader(
    #     dataset1, num_workers=0, pin_memory=True, shuffle=True,
    #     batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset2, num_workers=0, pin_memory=True, shuffle=True,
    #     batch_size=batch_size)

    ## Training
    # model = Net()

    loss_acc_dict = {}
    for solver in solvers_opts.keys():
        params = solvers_opts[solver]
        ## Data
        train_loader = torch.utils.data.DataLoader(
            dataset1, num_workers=0, pin_memory=True, shuffle=True,
            batch_size=params[2])
        test_loader = torch.utils.data.DataLoader(
            dataset2, num_workers=0, pin_memory=True, shuffle=True,
            batch_size=params[2])
        ## Training
        # torch.manual_seed(42)
        model = Net()  # si dovrebbe proprio azzerare il modello che mi sa in qualche modo ha memoria
        print(solver)
        sequences = train_loop(
            train_loader, test_loader, model, device, params[0], params[1],
            False, max_epochs, False, False)
        loss_acc_dict[solver] = sequences[:2]

    # losses_train, accs_train, losses_test, accs_test = train_loop(
    #     train_loader, test_loader, model, device, opts.lr, opts.momentum,
    #     opts.nesterov, opts.max_epochs)

    ## Diagnostic
    multiple_diagnostic(loss_acc_dict)
    plt.savefig("./plots/es2_2.pdf")
    print("Plot saved")
    # plt.savefig(opts.save_to)
    # print(f"Saved to {opts.save_to}")


if __name__ == "__main__":

    # test learning rate and momentum
    # solvers_opts = {"SGD1": [0.01, 0., 64], "SGD2": [0.001, 0., 64],
    #                "SGDM1": [0.01, 0.9, 64], "SGDM2": [0.001, 0.9, 64]}
    # test batch size
    solvers_opts = {"SGD1": [0.01, 0., 32], "SGD2": [0.01, 0., 64],
                    "SGD3": [0.01, 0., 128]}

    ### If options are passed through a yaml file
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config", help="YAML Configuration file")

    # # opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)

    # # store parsed arguments into args for clarity
    # args = parser.parse_args()
    # with open(args.config, "r") as f:
    #     # ensures the file il properly closed afeter reading
    #     # SafeLoader is generally preferred
    #     opts = yaml.load(f, Loader=yaml.SafeLoader)

    ### If options are passed through a dictionary, use for debugging
    # opts = dict(lr=0.01, momentum=0., nesterov=False, batch_size=64, max_epochs=5)
    # opts = dict(lr=[0.01, 0.001], momentum=[0., 0.9], nesterov=False,
                # batch_size=64, max_epochs=5)

    # opts = SimpleNamespace(**opts)

    # with launch_ipdb_on_exception():
    main(solvers_opts)
