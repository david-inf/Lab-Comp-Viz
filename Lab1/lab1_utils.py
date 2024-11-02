"""
Utils function for getting through the lab

"""

import time
import numpy as np
import matplotlib.pyplot as plt

import json

import torch
import torch.optim as optim


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_trainable_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, device, train_loader, criterion, optimizer):
    model.train()  # training mode

    losses, accs = [], []
    correct = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        data, target = data.to(device), target.to(device)

        ## zero the parameter gradients
        optimizer.zero_grad()

        ## forward + backward + optimize
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        ## Prediction
        pred = output.argmax(dim=1, keepdim=True)
        correct = torch.eq(pred, target.view_as(pred)).float()
        acc = torch.mean(correct)

        losses.append(loss.detach().cpu().numpy())
        accs.append(acc.detach().cpu().numpy())

    loss_k = np.array(losses).mean()
    acc_k = np.array(accs).mean()

    return loss_k, acc_k


def test(model, device, criterion, test_loader):
    model.eval()  # evaluation mode

    losses, accs = [], []
    correct = 0.

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            ## Compute loss
            output = model(data)
            loss = criterion(output, target)

            ## Prediction
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = torch.eq(pred, target.view_as(pred)).float()
            # correct += pred.eq(target.view_as(pred)).sum().item()
            acc = torch.mean(correct)

            ## Update loss and accuracy
            losses.append(loss.detach().cpu().numpy())
            accs.append(acc.detach().cpu().numpy())

    loss_k = np.array(losses).mean()
    acc_k = np.array(accs).mean()

    return loss_k, acc_k


def train_loop(train_loader, test_loader, model, criterion, device,
               lr, momentum, max_epochs, do_test=True):
    """" Training loop with SGD """

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    losses_train, accs_train = [], []
    losses_test, accs_test = [], []
    
    _start = time.time()
    _epoch_time = time.time()

    for epoch in range(1, max_epochs + 1):
        loss_train, acc_train = train(model, device, train_loader, criterion, optimizer)
        print(f"Epoch: {epoch}, Learning rate: {get_lr(optimizer):.6f}")
        print(f"Training - Loss: {loss_train:.4f}, Accuracy: {acc_train:.2f}, Runtime: {(time.time() - _epoch_time):.2f}")
        losses_train.append(loss_train)
        accs_train.append(acc_train)

        if do_test:
            loss_test, acc_test = test(model, device, criterion, test_loader)
            losses_test.append(loss_test)
            accs_test.append(acc_test)
            print(f"Test - Loss: {loss_test:.4f}, Accuracy: {acc_test:.2f}")

        _epoch_time = time.time()

    _end = time.time()
    print(f"Done! - Runtime: {(_end-_start):.2f} seconds")

    # test_class(model, device, criterion, testloader)

    if do_test:
        return losses_train, accs_train, losses_test, accs_test
    else:
        return losses_train, accs_train


def train_loop_sched(train_loader, test_loader, model, criterion, device,
                     optimizer, scheduler=None, max_epochs=10, do_test=True):
    """ Training loop with custom optimizer and optional learning rate scheduler """

    losses_train, accs_train = [], []
    losses_test, accs_test = [], []

    _start = time.time()
    _epoch_time = time.time()

    # epoch number only for logging
    for epoch in range(1, max_epochs + 1):
        ## Training
        loss_train, acc_train = train(model, device, train_loader, criterion, optimizer)
        print(f"Epoch: {epoch}, Learning rate: {get_lr(optimizer):.6f}")
        print(f"Training - Loss: {loss_train:.4f}, Accuracy: {acc_train:.3f}, Runtime: {(time.time() - _epoch_time):.2f}")
        losses_train.append(loss_train)
        accs_train.append(acc_train)
        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        ## Test/Validation
        if do_test:
            loss_test, acc_test = test(model, device, criterion, test_loader)
            losses_test.append(loss_test)
            accs_test.append(acc_test)
            print(f"Test - Loss: {loss_test:.4f}, Accuracy: {acc_test:.3f}")

        _epoch_time = time.time()

    _end = time.time()
    print(f"Done! - Runtime: {(_end-_start):.2f} seconds")

    if do_test:
        return losses_train, accs_train, losses_test, accs_test
    else:
        return losses_train, accs_train


def test_class(model, device, criterion, test_loader, classes):
    model.eval()  # configura il modello in evaluation mode

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # prepare to count overall predictions
    losses, accs = [], []
    correct = 0.

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            ## Fit data and compute loss
            output = model(data)
            loss = criterion(output, target)

            ## Prediction
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # compute overall accuracy
            correct = torch.eq(pred, target.view_as(pred)).float()
            acc = torch.mean(correct)
            # update loss and accuracy
            losses.append(loss.detach().cpu().numpy())
            accs.append(acc.detach().cpu().numpy())

            for target_i, pred_i in zip(target, pred):
                if target_i == pred_i:
                    correct_pred[classes[target_i]] += 1
                total_pred[classes[target_i]] += 1

    loss_final = np.array(losses).mean()
    acc_final = np.array(accs).mean()
    print(f"Final loss: {loss_final:.4f}, Accuracy: {acc_final:.3f}")
    print("-------")

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def simple_diagnostic(max_epochs, losses_train, accs_train):
    epochs_seq = np.arange(1, max_epochs + 1)
    # epoch_labels = 

    # plot only training loss and accuracy
    fig, ax = plt.subplots()
    # fig.suptitle("Training performance")
    fig.suptitle("Training loss and accuracy againts epochs")

    color = "tab:blue"
    ax.plot(epochs_seq, losses_train, label="loss", color=color)
    # ax.set_title("Training loss and accuracy againts epochs")
    ax.grid("both")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss", color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.set_xticks(np.arange(1, max_epochs+1, step=2))
    ax.set_xticklabels(np.arange(1, max_epochs + 1, 2))

    color = "tab:red"
    ax_1 = ax.twinx()
    ax_1.plot(epochs_seq, accs_train, label="accuracy", color="tab:red")
    ax_1.set_ylabel("Accuracy", color="tab:red")
    ax_1.tick_params(axis="y", labelcolor=color)


def diagnostic(losses_train, accs_train, losses_test, accs_test):
    ## left side training performance
    ## right side test performance
    max_epochs = len(losses_train)
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


def multiple_diagnostic_single(loss_acc_dict, max_epochs=10):
    # loss_acc_dict = {"Solver1": [loss, acc]...}
    epochs_seq = np.arange(1, max_epochs + 1)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    fig, ax = plt.subplots()
    ax_1 = ax.twinx()
    # fig.suptitle("CNN training performance over CIFAR10")
    fig.suptitle("Training loss and accuracy againts epochs")

    for i, (solver_name, perf) in enumerate(loss_acc_dict.items()):

        color = colors[i]
        ## plot loss function performance
        ax.plot(epochs_seq, perf[0], label=solver_name, color=color)
        ax.grid("both")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.tick_params(axis="y")
        ax.set_xticks(np.arange(1, max_epochs+1, step=2))
        ax.set_xticklabels(np.arange(1, max_epochs + 1, 2))

        ## plot accuracy performance
        ax_1.plot(epochs_seq, perf[1], label=solver_name, color=color)
        ax_1.set_ylabel("Accuracy")
        ax_1.tick_params(axis="y")

    ax.legend()


def multiple_diagnostic(loss_acc_dict, max_epochs=None, title_left="Training loss against epochs",
                        title_right="Test accuracy against epochs", fig_title=""):
    # loss_acc_dict = {"Solver1": [loss, acc]...}

    if max_epochs is None:
        max_epochs = len(next(iter(loss_acc_dict.values()))[0])
        # print(max_epochs)

    epochs_seq = np.arange(1, max_epochs + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fig.suptitle(fig_title)

    for solver_name, perf in loss_acc_dict.items():

        # plot loss function performance
        axs[0].plot(epochs_seq, perf[0], label=solver_name)
        axs[0].grid("both")
        axs[0].set_title(title_left)
        # axs[0].set_xlabel("Epochs")
        # axs[0].set_ylabel("Loss")
        axs[0].tick_params(axis="y")
        axs[0].set_xticks(np.arange(1, max_epochs+1, step=2))
        axs[0].set_xticklabels(np.arange(1, max_epochs+1, 2))

        # plot accuracy performance
        axs[1].plot(epochs_seq, perf[1], label=solver_name)
        axs[1].grid("both")
        axs[1].set_title(title_right)
        # axs[1].set_xlabel("Epochs")
        # axs[1].set_ylabel("Accuracy")
        axs[1].tick_params(axis="y")
        axs[1].set_xticks(np.arange(1, max_epochs+1, step=2))
        axs[1].set_xticklabels(np.arange(1, max_epochs+1, 2))

    axs[0].legend()
    axs[1].legend()


def save_to_json(loss_acc_dict, file_name):
    """ Save diagnostic to JSON file for reuse """
    # loss_acc_dict: {"Solver1": [[list of np.float32], [list of np.float32]]}
    # file_name: "file_name.json"

    # convert to float for serialization
    dict_converted = {k: [[float(x) for x in seq] for seq in v] for k, v in loss_acc_dict.items()}

    with open(file_name, "w") as file:
        json.dump(dict_converted, file)
