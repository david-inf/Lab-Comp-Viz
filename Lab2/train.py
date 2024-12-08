import time
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import wandb


def get_value(x):
    # detach from computational graph
    # send to cpu
    # convert to numpy
    return x.detach().cpu().numpy()


def train_log(epoch, batch_ct, train_loss):
    """ Log training metrics to WandB """
    log_dict = {
        "epoch": epoch,
        "train loss": train_loss,
    }
    wandb.log(log_dict, step=batch_ct)


def val_log(epoch, batch_ct, val_acc):
    """ Log validation metrics to WandB """
    log_dict = {
        "epoch": epoch, "val accuracy": val_acc
    }
    wandb.log(log_dict, step=batch_ct)


def linear_probe(model, in_features, train_loader, config):

    # add linear layer
    model.fc = nn.Linear(in_features, 10)
    model = model.to(config.device)
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # let final layer be trainable
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.eval_lr,
        momentum=config.eval_momentum,
        nesterov=config.eval_nesterov,
        weight_decay=config.eval_decay
    )

    epoch, loss_train, acc_train = train_loop_eval(
        model, train_loader, optimizer, criterion, config)

    return epoch, loss_train, acc_train


def test(model, test_loader, config):
    """" Test loop for classification model """
    model.eval()  # evaluation mode

    # loss_fn = torch.nn.CrossEntropyLoss()
    # losses, accs = [], []
    correct = []  # will be of the dataset size

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for X, y in tepoch:
                tepoch.set_description("Testing")

                X, y = X.to(config.device), y.to(config.device)
                output = model(X)  # [batch_size, num_classes]
                pred = np.argmax(get_value(output), axis=1)  # (batch_size,)
                # loss = loss_fn(output, y)  # scalar

                # losses.append(get_value(loss))
                correct_i = pred == get_value(y)  # boolean array
                correct.extend(list(correct_i))

    return np.mean(correct)


def train_loop_ssl(model, train_loader, criterion, optimizer, config):
    """ Training loop for classification model """
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    step = 0  # logging
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (view1, view2, _) in enumerate(tepoch):
                tepoch.set_description(f"Train epoch {epoch}")

                ## -----
                view1, view2 = view1.to(config.device), view2.to(config.device)
                optimizer.zero_grad()  # zero the gradients
                # Forward pass -> logits
                output = model(view1, view2)
                features1, features2 = output.values()
                loss = criterion(features1, features2)
                # Backward pass
                loss.backward()
                optimizer.step()
                # Metrics
                losses.append(get_value(loss))
                ## -----

                # Log metrics every log_every batches
                if batch_idx % config.log_every == 0:
                    train_loss = np.mean(losses[-config.batch_window:])
                    # log to wandb
                    train_log(epoch, batch_idx, train_loss)
                    # log to console
                    tepoch.set_postfix(loss=train_loss)
                    tepoch.update()
                    step += 1

    return epoch, train_loss


def train_loop_eval(model, train_loader, optimizer, criterion, config):

    step = 0
    for epoch in range(1, config.eval_epochs + 1):
        model.train()
        losses, accs = [], []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (X, y) in enumerate(tepoch):
                tepoch.set_description(f"Train epoch {epoch}")

                ## -----
                X, y = X.to(config.device), y.to(config.device)
                optimizer.zero_grad()  # zero the gradients
                # Forward pass -> logits
                output = model(X)
                pred = np.argmax(get_value(output), axis=1)
                loss = criterion(output, y)
                # Backward pass
                loss.backward()
                optimizer.step()
                # Metrics
                losses.append(get_value(loss))
                accs.append(np.mean(pred == get_value(y)))
                ## -----

                if batch_idx % config.log_every == 0:
                    train_loss = np.mean(losses[-config.batch_window:])
                    train_acc = np.mean(accs[-config.batch_window:])
                    tepoch.set_postfix(loss=train_loss, accuracy=100.*train_acc)
                    tepoch.update()
                    step += 1
        
        # TODO: put here validation set

    return epoch, losses[-1], accs[-1]
