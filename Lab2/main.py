import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, efficientnet_b0

import wandb

from train import train_loop_ssl, train_loop_eval, test
from dataset import BaseDataset, AugmentedImageDataset, MakeDataLoaders
from nets import BTNet


def ssl_train(config):
    from torch.utils.data import DataLoader
    from nets import SiameseNetSync as SiameseNet
    from barlow import BarlowTwins

    ### Dataset and DataLoader
    trainset = AugmentedImageDataset()
    train_loader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )

    ### Model and Loss function
    backbone = None
    if config.backbone == "ResNet18":
        backbone = resnet18()
    elif config.backbone == "EfficientNet-B0":
        backbone = efficientnet_b0()

    model = SiameseNet(backbone)
    model = model.to(config.device)
    criterion = BarlowTwins(0.005)
    criterion = criterion.to(config.device)

    ### SSL Optimizer
    params = [{
        "params": model.parameters(),  # backbone
        "params": criterion.projector.parameters(),  # projector
        "params": criterion.bn.parameters()  # batch norm layer
    }]
    optimizer = optim.Adam(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    ### Run pre-training
    epoch, loss = train_loop_ssl(
        model, train_loader, criterion, optimizer, config
    )
    print("\n**** Pre-training done! ****\n")

    ### Save the model in the exchangeable ONNX format
    # input_data = torch.rand(1, 3, 32, 32).to(config.device)
    # torch.onnx.export(model, (input_data, input_data), "model.onnx")
    # wandb.save("model.onnx")
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, "./models/" + config.model_name + "_pre-train" + f"_e_{epoch}.pt")

    return backbone, criterion.projector


def sl_train(config, backbone, projector):

    ### Linear probe training
    train_data = BaseDataset()
    testset = BaseDataset(train=False)
    loader = MakeDataLoaders(train_data, testset, config)
    train_loader = loader.train_loader
    # val_loader = loader.val_loader
    test_loader = loader.test_loader

    ### Model (linear probing)
    repr_dim = projector[-1].out_features  # representation dimension
    layers = [
        nn.BatchNorm1d(repr_dim),
        nn.ReLU(inplace=True),
        nn.Linear(repr_dim, 10)
    ]
    classifier = nn.Sequential(*layers)
    # freeze all layers
    for param in backbone.parameters():
        param.requires_grad = False
    for param in projector.parameters():
        param.requires_grad = False
    # final model
    model = BTNet(backbone, projector, classifier)
    model = model.to(config.device)

    ### Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.eval_lr,
        momentum=config.eval_momentum,
        nesterov=config.eval_nesterov,
        weight_decay=config.eval_decay
    )

    ### Run supervised training
    epoch, loss_train, acc_train = train_loop_eval(
        model, train_loader, optimizer, criterion, config
    )
    print("\n**** Supervised learning training done! ****\n")

    ### Test model
    test_acc = test(
        model, test_loader, config
    )
    print(f"\n**** Testing done - accuracy: {test_acc} ****\n")
    wandb.log({"test accuracy": test_acc})

    ### Save the model in the exchangeable ONNX format
    # input_data = torch.rand(1, 3, 32, 32).to(config.device)
    # torch.onnx.export(model, (input_data, input_data), "model.onnx")
    # wandb.save("model.onnx")
    torch.save({
        "epoch": epoch,
        "train loss": loss_train,
        "train acc": acc_train,
        "test acc": test_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, "./models/" + config.model_name + "_eval" + f"_e_{epoch}.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML COnfiguration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.SafeLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["device"] = device


    ## **** Pre-training **** ##
    wandb_config = dict(
        project="cv-lab2",
        config=hyperparams,
        entity="alessiochen98-university-of-florence",
        name=hyperparams["model_name"] + "_pretrain"
    )

    with wandb.init(**wandb_config):
        config = wandb.config
        backbone, projector = ssl_train(config)


    ## **** Training **** ##
    wandb_config = dict(
        project="cv-lab2",
        config=hyperparams,
        entity="alessiochen98-university-of-florence",
        name=hyperparams["model_name"] + "_eval"
    )

    with wandb.init(**wandb_config):
        config = wandb.config
        sl_train(config, backbone, projector)
