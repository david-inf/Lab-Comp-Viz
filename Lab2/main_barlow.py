# from types import SimpleNamespace
import argparse
import yaml
# import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR#, ReduceLROnPlateau
from torchvision.models import resnet18#, efficientnet_b0

import wandb

from nets import SimpleNet
from train import train_loop_ssl, train_loop_eval, test
from dataset import BaseDataset, AugmentedImageDataset, MakeDataLoaders, make_loader


def ssl_train(config):
    from nets import SiameseNetSync as SiameseNet
    from barlow import BarlowTwins

    ### Dataset and DataLoader
    trainset = AugmentedImageDataset()  # training set, Dataset object
    train_loader = make_loader(trainset, config)  # DataLoader object

    ### Model and Loss function
    print("**** Checking backbone", config.backbone, "****")
    backbone = None  # load the backbone
    if config.backbone == "ResNet18":
        backbone = resnet18(weights="DEFAULT")  # !this one!
    # elif config.backbone == "EfficientNet-B0":
    #     backbone = efficientnet_b0()  # needs to adapt code
    elif config.backbone == "SimpleNet":
        backbone = SimpleNet(16, config.z_dim, 10)

    print("**** Loading model to", config.device, "****")
    model = SiameseNet(backbone)  # synchronized siamese network
    model = model.to(config.device)
    criterion = BarlowTwins(0.005, config.z_dim)  # computes the loss
    criterion = criterion.to(config.device)

    ### SSL Optimizer
    params = [  # list of dict
        # backbone with global learning rate
        {"params": model.parameters()},
        # projector with different starting learning rate
        {"params": criterion.projector.parameters(), "lr": 1e-2},
        # batch norm layer with different starting learning rate
        {"params": criterion.bn.parameters(), "lr": 1e-2}
    ]
    optimizer = optim.SGD(
        params,
        lr=config.learning_rate,  # global settings
        momentum=config.momentum,
        nesterov=config.nesterov,
        weight_decay=config.weight_decay
    )
    # scheduler = ReduceLROnPlateau(
    #     optimizer, "min",
    #     threshold=1e-1, patience=4)
    scheduler = ExponentialLR(
        optimizer,
        gamma=0.9,  # lr * gamma**k
    )

    ### Run pre-training
    print("\n**** Starting pre-training ****")
    epoch, loss = train_loop_ssl(
        model, train_loader, criterion, optimizer, scheduler, config
    )
    print("**** Pre-training done! ****\n")

    ### TODO: Save the model in the exchangeable ONNX format
    # input_data = torch.rand(1, 3, 32, 32).to(config.device)
    # torch.onnx.export(model, (input_data, input_data), "model.onnx")
    # wandb.save("model.onnx")
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, "./models/" + config.model_name + "_pre-train" + f"_e_{epoch}.pt")

    return backbone


def sl_train(config, backbone):
    from barlow import BTNet

    ### Dataset and DataLoader
    trainset = BaseDataset()  # training set, Dataset object
    train_loader = make_loader(trainset, config)  # DataLoader object
    testset = BaseDataset(train=False)  # test set, Dataset object
    test_loader = make_loader(testset, config)  # DataLoader object

    # loader = MakeDataLoaders(train_data, testset, config)
    # train_loader = loader.train_loader
    # TODO: val_loader = loader.val_loader
    # test_loader = loader.test_loader

    ### Model - evaluation
    repr_dim = config.z_dim  # latent representation size
    if config.eval == "linear":
        layers = [
            nn.Linear(repr_dim, 10)  # logits
        ]
    elif config.eval == "MLP":
        layers = [
            nn.Linear(repr_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)  # logits
        ]
    classifier = nn.Sequential(*layers)

    # freeze backbone
    print("**** Freezing layers ****")
    for param in backbone.parameters():
        param.requires_grad = False

    # final model
    print("**** Loading model to", config.device, "****")
    model = BTNet(backbone, classifier)
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

    ### Test model -> test accuracy
    test_acc = test(
        model, test_loader, config
    )
    print(f"\n**** Testing done - accuracy: {test_acc} ****\n")
    wandb.log({"test accuracy": test_acc})

    ### TODO: Save the model in the exchangeable ONNX format
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
    # project = "comp-viz"
    # entity = "david-inf-team"

    ## gruppo lab2
    project = "cv-lab2"
    entity = "alessiochen98-university-of-florence"

    def merge_configs(global_config, task_config):
        merged = global_config.copy()
        merged.update(task_config)
        return merged

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML COnfiguration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        # get hyper-parameters as dict
        hp = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    device = "cuda" if torch.cuda.is_available() else "cpu"  # check device

    # create two different configs for pre-training and downstream task
    global_hp = hp["global"]  # global configs
    global_hp["device"] = device
    pre_hp = hp["tasks"]["pretraining"]  # pre-training configs
    down_hp = hp["tasks"]["downstream"]  # downstream task configs
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
        config = wandb.config  # class with pre-training configs as attributes
        backbone = ssl_train(config)  # run training, return encoder

    ## **** Downstream task - evaluation **** ##
    wandb_config = dict(
        project=project,
        config=down_config,
        entity=entity,
        name=global_hp["model_name"] + "_eval"
    )
    with wandb.init(**wandb_config):
        config = wandb.config  # class with downstream task configs as attributes
        sl_train(config, backbone)  # run training 
