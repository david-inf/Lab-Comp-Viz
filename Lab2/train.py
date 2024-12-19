# import time
import numpy as np

from tqdm import tqdm

import torch

import wandb


def get_value(x):
    # detach from computational graph
    # send to cpu
    # convert to numpy
    return x.detach().cpu().numpy()


def train_loop_ssl(model, train_loader, criterion, optimizer, scheduler, config):
    """ Training loop for SSL pre-training """
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    step = 0  # logging step
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (view1, view2, _) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                ## -----
                view1, view2 = view1.to(config.device), view2.to(config.device)
                optimizer.zero_grad()  # zero the gradients
                # Forward pass -> logits
                output = model(view1, view2)  # [N, z_dim], [N, z_dim]
                features1, features2 = output.values()
                loss = criterion(features1, features2)
                # Backward pass
                loss.backward()
                optimizer.step()
                # Metrics
                losses.append(get_value(loss))
                ## -----

                # Log metrics every `log_every` batches and take loss for `batch_window`
                if batch_idx % config.log_every == 0:
                    ## Pre-training metrics
                    train_loss = np.mean(losses[-config.batch_window:])
                    ## Log SSL metrics to wandb
                    wandb.log({
                        "epoch": epoch,
                        "pre-train loss": train_loss
                    }, step=step)
                    ## Log to console
                    tepoch.set_postfix(
                        loss=train_loss,
                        lr=scheduler.get_last_lr()
                    )
                    tepoch.update()
                    step += 1

            # scheduler.step(np.mean(losses))  # if plateau
            scheduler.step()  # if exponential

    return epoch, train_loss


def train_loop_eval(model, train_loader, optimizer, criterion, config, val_loader=None):
    """ Training loop for SL training that can be used for SSL evaluation """
    wandb.watch(model, criterion, log="all", log_freq=10)

    step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses, accs = [], []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (X, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

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
                    ## Training metrics
                    train_loss = np.mean(losses[-config.batch_window:])
                    train_acc = np.mean(accs[-config.batch_window:])
                    ## TODO: Validation metrics
                    # val_acc = test(model, val_loader, config)
                    ## Log SL metrics to wandb
                    wandb.log({
                        "epoch": epoch,
                        "train loss": train_loss,
                        "train accuracy": train_acc,
                        # "val accuracy": val_acc
                    }, step=step)
                    ## Log to console
                    tepoch.set_postfix(loss=train_loss, acc=100.*train_acc)
                    tepoch.update()
                    step += 1

    return epoch, losses[-1], accs[-1]


def test(model, test_loader, config):
    """" Test/Validation loop for SL training """
    model.eval()  # evaluation mode

    correct = []  # will be of the dataset size
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for X, y in tepoch:
                tepoch.set_description("Testing")

                ## -----
                X, y = X.to(config.device), y.to(config.device)
                output = model(X)  # [batch_size, num_classes]
                pred = np.argmax(get_value(output), axis=1)  # (batch_size,)
                # loss = loss_fn(output, y)  # scalar

                # losses.append(get_value(loss))
                correct_i = pred == get_value(y)  # boolean array
                correct.extend(list(correct_i))
                ## -----

    return np.mean(correct)
