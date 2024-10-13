# -*- coding: utf-8 -*-
"""
Utils functions

"""

import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# %% Training loop

# TODO: different levels of logging
def train(model, device, train_loader, optimizer, epoch, info=True):
    # ogni layer può avere un comportamento a seconda di training o test
    # esempio è il dropout, batch normalization
    model.train()  # configura il modello in training mode

    losses, accs = [], []
    correct = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        ## Make a step
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)  # compute loss
        loss.backward()  # compute gradient
        optimizer.step()  # update weights

        ## Prediction
        pred = output.argmax(dim=1, keepdim=True)
        correct = torch.eq(pred, target.view_as(pred)).float()
        acc = torch.mean(correct)

        ## Update loss and accuracy
        losses.append(loss.detach().cpu().numpy())
        # mi sa anche loss.item()
        accs.append(acc.detach().cpu().numpy())

        # if batch_idx % 500 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        if info:
            if batch_idx % 500 == 0:
                print(f"Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]",
                      f"Loss: {loss.item():.4f}",
                      f"Accuracy: {100 * acc:.1f}%"
                      )

    loss_k = np.array(losses).mean()
    acc_k = np.array(accs).mean()

    return loss_k, acc_k


def test(model, device, test_loader, info=True):
    model.eval()  # configura il modello in evaluation mode

    losses, accs = [], []
    correct = 0.

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            ## Compute loss
            output = model(data)
            loss = F.nll_loss(output, target)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            ## Prediction
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = torch.eq(pred, target.view_as(pred)).float()
            # correct += pred.eq(target.view_as(pred)).sum().item()
            acc = torch.mean(correct)

            ## Update loss and accuracy
            losses.append(loss.detach().cpu().numpy())
            accs.append(acc.detach().cpu().numpy())


    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    # test_loss /= len(test_loader.dataset)
    # accuracy /= len(test_loader.dataset)
    loss_k = np.array(losses).mean()
    acc_k = np.array(accs).mean()

    if info:
        # after each epoch
        print(f"Test set - Loss: {loss_k:.4f} - Accuracy: {100 * acc_k:.1f}%")

    return loss_k, acc_k


def train_loop(train_loader, test_loader, model, device, lr, momentum,
               nesterov, max_epochs, info=True, do_test=True):

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          nesterov=nesterov)
    
    losses_train, accs_train = [], []
    losses_test, accs_test = [], []
    
    epochs = max_epochs
    _start = time.time()
    for epoch in range(1, epochs + 1):
        loss_train, acc_train = train(model, device, train_loader, optimizer,
                                      epoch, info)
        losses_train.append(loss_train)
        accs_train.append(acc_train)

        if do_test:
            loss_test, acc_test = test(model, device, test_loader, info)
            losses_test.append(loss_test)
            accs_test.append(acc_test)
    _end = time.time()
    print(f"Done! - Runtime: {(_end-_start):.2f} seconds")

    return losses_train, accs_train, losses_test, accs_test


# %% Training diagnostic

def diagnostic(max_epochs, losses_train, accs_train, losses_test, accs_test):
    epochs_seq = np.arange(1, max_epochs + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fig.suptitle("CNN performance over MNIST")

    ## 1) train loss (first y axis) and accuracy (second y axis)
    color = "tab:blue"
    axs[0].plot(epochs_seq, losses_train, label="loss", color=color)
    axs[0].set_title("Training loss and accuracy againts epochs")
    axs[0].grid("both")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss", color=color)
    axs[0].tick_params(axis="y", labelcolor=color)

    color = "tab:red"
    axs0_1 = axs[0].twinx()
    axs0_1.plot(epochs_seq, accs_train, label="accuracy", color="tab:red")
    axs0_1.set_ylabel("Accuracy", color="tab:red")
    axs0_1.tick_params(axis="y", labelcolor=color)

    ## 2) test loss (first y axis) and accuracy (second y axis)
    color = "tab:blue"
    axs[1].plot(epochs_seq, losses_test, label="loss", color=color)
    axs[1].set_title("Test loss and accuracy againts epochs")
    axs[1].grid("both")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss", color=color)
    axs[1].tick_params(axis="y", labelcolor=color)

    color = "tab:red"
    axs1_1 = axs[1].twinx()
    axs1_1.plot(epochs_seq, accs_test, label="accuracy", color="tab:red")
    axs1_1.set_ylabel("Accuracy", color="tab:red")
    axs1_1.tick_params(axis="y", labelcolor=color)

    # plt.show()


def multiple_diagnostic(loss_acc_dict):
    # loss_acc_dict = {"Solver1": [loss, acc]...}

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fig.suptitle("CNN training performance over MNIST")

    for solver_name, perf in loss_acc_dict.items():

        # plot loss function performance
        axs[0].plot(perf[0], label=solver_name)
        axs[0].grid("both")
        axs[0].set_title("Loss against epochs")

        # plot accuracy performance
        axs[1].plot(perf[1], label=solver_name)
        axs[1].grid("both")
        axs[1].set_title("Accuracy against epochs")

    axs[0].legend()
    axs[1].legend()
