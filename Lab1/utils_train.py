import time
import numpy as np

import torch
import torch.optim as optim


def get_value(x):
    # detach from computational graph
    # send to cpu
    # convert to numpy
    return x.detach().cpu().numpy()


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_metrics(model, test_loader, opts):
    """"
    Test loop for classification model
    Compute metrics by distributed computing through batches
    """
    model.eval()  # evaluation mode

    # loss_fn = torch.nn.CrossEntropyLoss()
    # losses, accs = [], []
    correct = []  # will be of the dataset size

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(opts.device), y.to(opts.device)
            ## Forward pass -> logits
            output = model(X)  # [batch_size, num_classes]
            pred = np.argmax(get_value(output), axis=1)  # (batch_size,)
            # loss = loss_fn(output, y)  # scalar
            ## Save metrics
            # losses.append(get_value(loss))
            # accs.append(np.mean(pred == get_value(y)))
            correct_i = pred == get_value(y)  # boolean array
            correct.extend(list(correct_i))

    return np.mean(correct)


def train_loop(model, train_loader, val_loader,
               optimizer, opts):
    """
    Training loop for classification model
    """

    import tensorflow as tf
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/Train')
    val_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/Val')

    # Cross Entropy Loss, reduction="mean"
    loss_fn = torch.nn.CrossEntropyLoss()  # (logits, actual_class)

    _start = time.time()
    _epoch_time = time.time()
    # step = 0

    for epoch in range(1, opts.max_epochs + 1):
        model.train()  # training mode
        # this holds only if the batches are of the same size! (the last one may excepts)
        losses, accs = [], []  # loss and accuracy for each batch
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(opts.device), y.to(opts.device)

            ### Model update
            optimizer.zero_grad()  # zero the gradients
            ## Forward pass -> logits
            output = model(X)  # [batch_size, num_classes]
            pred = np.argmax(get_value(output), axis=1)  # (batch_size,)
            loss = loss_fn(output, y)  # returns a scalar value
            ## Backward pass
            loss.backward()
            optimizer.step()

            ### Save metrics
            losses.append(get_value(loss))
            accs.append(np.mean(pred == get_value(y)))

            if batch_idx % opts.log_every == 0:
                # training loss and accuracy
                train_loss = np.mean(losses)
                train_acc = np.mean(accs)
                # validation accuracy
                val_acc = test_metrics(model, val_loader, opts)

                print("Epoch [%d/%d][%d/%d] | Train loss: %.4f accuracy: %.3f | Val accuracy: %.3f"
                      % (epoch, opts.max_epochs, batch_idx, len(train_loader),
                         train_loss, train_acc, val_acc))

        with train_writer.as_default():
            tf.summary.scalar('Loss', train_loss, step=epoch)
            tf.summary.scalar('Accuracy', train_acc, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('Accuracy', val_acc, step=epoch)

        # step += 1

        print("Epoch %d - %.2f seconds" % (epoch, time.time()-_epoch_time))
        _epoch_time = time.time()

    print("Done! %.2f seconds" % (time.time()-_start))


# def train(model, device, train_loader, criterion, optimizer):
#     """ Train one epoch """
#     model.train()  # training mode

#     losses, accs = [], []
#     correct = 0.

#     for _, (data, target) in enumerate(train_loader):
#         # get the inputs; data is a list of [inputs, labels]
#         data, target = data.to(device), target.to(device)

#         ## zero the parameter gradients
#         optimizer.zero_grad()

#         ## forward + backward + optimize
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#         ## Prediction
#         pred = output.argmax(dim=1, keepdim=True)
#         correct = torch.eq(pred, target.view_as(pred)).float()
#         acc = torch.mean(correct)

#         losses.append(loss.detach().cpu().numpy())
#         accs.append(acc.detach().cpu().numpy())

#     # loss and accuracy for this epoch
#     loss_k = np.array(losses).mean()
#     acc_k = np.array(accs).mean()

#     return loss_k, acc_k


# def test(model, device, criterion, test_loader):
#     model.eval()  # evaluation mode

#     losses, accs = [], []
#     correct = 0.

#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)

#             ## Compute loss
#             output = model(data)
#             loss = criterion(output, target)

#             ## Prediction
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct = torch.eq(pred, target.view_as(pred)).float()
#             # correct += pred.eq(target.view_as(pred)).sum().item()
#             acc = torch.mean(correct)

#             ## Update loss and accuracy
#             losses.append(loss.detach().cpu().numpy())
#             accs.append(acc.detach().cpu().numpy())

#     loss_k = np.array(losses).mean()
#     acc_k = np.array(accs).mean()

#     return loss_k, acc_k


# def train_loop(train_loader, test_loader, model, criterion, device,
#                lr, momentum, max_epochs, do_test=True):
#     """" Training loop with SGD """

#     model.to(device)

#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#     losses_train, accs_train = [], []
#     losses_test, accs_test = [], []
    
#     _start = time.time()
#     _epoch_time = time.time()

#     for epoch in range(1, max_epochs + 1):
#         loss_train, acc_train = train(model, device, train_loader, criterion, optimizer)
#         print(f"Epoch: {epoch}, Learning rate: {get_lr(optimizer):.6f}")
#         print(f"Training - Loss: {loss_train:.4f}, Accuracy: {acc_train:.2f}, Runtime: {(time.time() - _epoch_time):.2f}")
#         losses_train.append(loss_train)
#         accs_train.append(acc_train)

#         if do_test:
#             loss_test, acc_test = test(model, device, criterion, test_loader)
#             losses_test.append(loss_test)
#             accs_test.append(acc_test)
#             print(f"Test - Loss: {loss_test:.4f}, Accuracy: {acc_test:.2f}")

#         _epoch_time = time.time()

#     _end = time.time()
#     print(f"Done! - Runtime: {(_end-_start):.2f} seconds")

#     # test_class(model, device, criterion, testloader)

#     if do_test:
#         return losses_train, accs_train, losses_test, accs_test
#     else:
#         return losses_train, accs_train


# def train_loop_sched(train_loader, test_loader, model, criterion, device,
#                      optimizer, scheduler=None, max_epochs=10, do_test=True):
#     """ Training loop with custom optimizer and optional learning rate scheduler """

#     losses_train, accs_train = [], []
#     losses_test, accs_test = [], []

#     _start = time.time()
#     _epoch_time = time.time()

#     # epoch number only for logging
#     for epoch in range(1, max_epochs + 1):
#         ## Training
#         loss_train, acc_train = train(model, device, train_loader, criterion, optimizer)
#         print(f"Epoch: {epoch}, Learning rate: {get_lr(optimizer):.6f}")
#         print(f"Training - Loss: {loss_train:.4f}, Accuracy: {acc_train:.3f}, Runtime: {(time.time() - _epoch_time):.2f}")
#         losses_train.append(loss_train)
#         accs_train.append(acc_train)
#         # Learning rate scheduler
#         if scheduler is not None:
#             scheduler.step()
#         ## Test/Validation
#         if do_test:
#             loss_test, acc_test = test(model, device, criterion, test_loader)
#             losses_test.append(loss_test)
#             accs_test.append(acc_test)
#             print(f"Test - Loss: {loss_test:.4f}, Accuracy: {acc_test:.3f}")

#         _epoch_time = time.time()

#     _end = time.time()
#     print(f"Done! - Runtime: {(_end-_start):.2f} seconds")

#     if do_test:
#         return losses_train, accs_train, losses_test, accs_test
#     else:
#         return losses_train, accs_train


# def test_class(model, device, criterion, test_loader, classes):
#     model.eval()  # configura il modello in evaluation mode

#     # prepare to count predictions for each class
#     correct_pred = {classname: 0 for classname in classes}
#     total_pred = {classname: 0 for classname in classes}

#     # prepare to count overall predictions
#     losses, accs = [], []
#     correct = 0.

#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)

#             ## Fit data and compute loss
#             output = model(data)
#             loss = criterion(output, target)

#             ## Prediction
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

#             # compute overall accuracy
#             correct = torch.eq(pred, target.view_as(pred)).float()
#             acc = torch.mean(correct)
#             # update loss and accuracy
#             losses.append(loss.detach().cpu().numpy())
#             accs.append(acc.detach().cpu().numpy())

#             for target_i, pred_i in zip(target, pred):
#                 if target_i == pred_i:
#                     correct_pred[classes[target_i]] += 1
#                 total_pred[classes[target_i]] += 1

#     loss_final = np.array(losses).mean()
#     acc_final = np.array(accs).mean()
#     print(f"Final loss: {loss_final:.4f}, Accuracy: {acc_final:.3f}")
#     print("-------")

#     # print accuracy for each class
#     for classname, correct_count in correct_pred.items():
#         accuracy = 100 * float(correct_count) / total_pred[classname]
#         print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
