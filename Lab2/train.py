import time
import numpy as np

from tqdm import tqdm

import torch
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


# def test(model, test_loader, config):
#     """" Test loop for classification model """
#     model.eval()  # evaluation mode

#     # loss_fn = torch.nn.CrossEntropyLoss()
#     # losses, accs = [], []
#     correct = []  # will be of the dataset size

#     with torch.no_grad():
#         with tqdm(test_loader, unit="batch") as tepoch:
#             for X, y in tepoch:
#                 tepoch.set_description("Testing")

#                 X, y = X.to(config.device), y.to(config.device)
#                 output = model(X)  # [batch_size, num_classes]
#                 pred = np.argmax(get_value(output), axis=1)  # (batch_size,)
#                 # loss = loss_fn(output, y)  # scalar

#                 # losses.append(get_value(loss))
#                 correct_i = pred == get_value(y)  # boolean array
#                 correct.extend(list(correct_i))

#     return np.mean(correct)


def train_loop(model, train_loader, criterion, optimizer, config):
    """ Training loop for classification model """
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    batch_ct = 0  # count batches for logging to wandb
    for epoch in range(1, config.epochs + 1):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for view1, view2, _ in tepoch:
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
                train_loss = get_value(loss)
                ## -----

                batch_ct += 1  # batch processed

                # Log metrics every log_every batches
                if batch_ct % config.log_every == 0:
                    # log to wandb
                    # TODO: log the mean, so keep a list
                    train_log(epoch, batch_ct, train_loss)
                    # log to console
                    tepoch.set_postfix(loss=train_loss)

        # TODO: put here validation, at the end of each epoch
