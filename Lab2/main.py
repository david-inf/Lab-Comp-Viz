from types import SimpleNamespace
import argparse
import yaml
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18, efficientnet_b0
from torch.utils.data import DataLoader

import wandb

from nets import SimpleNet
from train import train_loop_ssl, train_loop_eval, test
from dataset import BaseDataset, AugmentedImageDataset, MakeDataLoaders
from nets import BTNet, SiameseNetSync as SiameseNet
from barlow import BarlowTwins


def ssl_train(config):

    ### Dataset and DataLoader
    trainset = AugmentedImageDataset()
    train_loader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )

    ### Model and Loss function
    print("**** Checking backbone", config.backbone, "****")
    backbone = None
    if config.backbone == "ResNet18":
        backbone = resnet18(weights="DEFAULT")
    elif config.backbone == "EfficientNet-B0":
        backbone = efficientnet_b0()
    elif config.backbone == "SimpleNet":
        backbone = SimpleNet(16, config.z_dim, 10)

    print("**** Loading model to", config.device, "****")
    model = SiameseNet(backbone)
    model = model.to(config.device)
    criterion = BarlowTwins(0.005, config.z_dim)
    criterion = criterion.to(config.device)

    ### SSL Optimizer
    params = [
        {"params": model.parameters()},  # backbone
        {"params": criterion.projector.parameters(), "lr": 1e-3},  # projector
        {"params": criterion.bn.parameters(), "lr": 1e-3}  # batch norm
    ]
    optimizer = optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov,
        weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "min",
        threshold=1e-2, patience=6)

    ### Run pre-training
    print("\n**** Starting pre-training ****")
    epoch, loss = train_loop_ssl(
        model, train_loader, criterion, optimizer, scheduler, config
    )
    print("**** Pre-training done! ****\n")

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

    ### Dataset and DataLoader
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
    print("**** Freezing layers ****")
    for param in backbone.parameters():
        param.requires_grad = False
    for param in projector.parameters():
        param.requires_grad = False
    # final model
    print("**** Loading model to", config.device, "****")
    # forse la parte di evaluation è soltanto con la backbone (encoder)
    model = BTNet(backbone, projector, classifier)
    model = model.to(config.device)

    ### Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov,
        weight_decay=config.weight_decay
    )

    ### Run supervised training
    print("**** Starting pre-training ****")
    epoch, loss_train, acc_train = train_loop_eval(
        model, train_loader, optimizer, criterion, config
    )
    print("**** Supervised learning training done! ****\n")

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

    ## mio personale
    project = "comp-viz"
    entity = "david-inf-team"

    ## gruppo lab2
    # project = "cv-lab2"
    # entity = "alessiochen98-university-of-florence"

    def merge_configs(global_config, task_config):
        merged = global_config.copy()
        merged.update(task_config)
        return merged

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML COnfiguration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        # get hyper-parameters
        hp = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    device = "cuda" if torch.cuda.is_available() else "cpu"

    global_hp = hp["global"]
    global_hp["device"] = device
    pre_hp = hp["tasks"]["pretraining"]
    down_hp = hp["tasks"]["downstream"]
    pre_config = merge_configs(global_hp, pre_hp)  # dict
    down_config = merge_configs(global_hp, down_hp)  # dict

    ## **** Pre-training **** ##
    wandb_config = dict(
        project=project,
        config=pre_config,
        entity=entity,
        name=global_hp["model_name"] + "_pretrain"
    )

    with wandb.init(**wandb_config):
        config = wandb.config
        backbone, projector = ssl_train(config)

    ## **** Training **** ##
    wandb_config = dict(
        project=project,
        config=down_config,
        entity=entity,
        name=global_hp["model_name"] + "_eval"
    )

    with wandb.init(**wandb_config):
        config = wandb.config
        sl_train(config, backbone, projector)
