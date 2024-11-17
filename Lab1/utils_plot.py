import numpy as np
import matplotlib.pyplot as plt
import json


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
