import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torchvision.models import resnet18, efficientnet_b0

import wandb

from train import train_loop_ssl, linear_probe, test
from dataset import BaseDataset, AugmentedImageDataset, MakeDataLoaders


def barlowtwins(config):
    from nets import SiameseNetSync as SiameseNet
    from barlow import BarlowTwins

    ## **** Pre-training **** ##
    ### Dataset and DataLoader
    trainset = AugmentedImageDataset()
    loader = MakeDataLoaders(trainset, config.batch_size)
    train_loader = loader.loader

    ### Model and Loss function
    backbone = None
    if config.backbone == "ResNet18":
        backbone = resnet18()
    elif config.backbone == "EfficientNet-B0":
        backbone = efficientnet_b0()

    model = SiameseNet(backbone)
    criterion = BarlowTwins(0.005)
    model = model.to(config.device)
    criterion = criterion.to(config.device)

    ### SSL Optimizer
    params = [{
        "params": model.parameters(),  # backbone
        "params": criterion.projector.parameters(),  # projector
        "params": criterion.bn.parameters()  # batch norm layer
    }]
    optimizer_ssl = optim.Adam(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    ### Run pre-training
    epoch_ssl, loss_ssl = train_loop_ssl(model, train_loader, criterion, optimizer_ssl, config)
    print("Pre-training done!")

    ## **** Evaluation **** ##
    ### Linear probe training
    trainset = BaseDataset()
    testset = BaseDataset(train=False)
    loader = MakeDataLoaders(trainset, config.batch_size)
    train_loader = loader.loader
    epoch_eval, loss_eval, acc_eval = linear_probe(backbone, model.in_features, train_loader, config)
    print("Training linear layer done!")

    ### Test model
    loader = MakeDataLoaders(testset, config.batch_size)
    test_loader = loader.loader
    test_acc = test(backbone, test_loader, config)
    print("Testing done - accuracy:", test_acc)
    wandb.log({"test_accuracy": test_acc})

    ### Save the model in the exchangeable ONNX format
    # input_data = torch.rand(1, 3, 32, 32).to(config.device)
    # torch.onnx.export(model, (input_data, input_data), "model.onnx")
    # wandb.save("model.onnx")
    torch.save({
        "epoch_ssl": epoch_ssl,
        "loss_ssl": loss_ssl,
        "epoch_eval": epoch_eval,
        "loss_eval": loss_eval,
        "acc_eval": acc_eval,
        "test_acc": test_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_ssl.state_dict()
    }, config.model_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML COnfiguration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.SafeLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["device"] = device

    wandb_config = dict(
        project="cv-lab2",
        config=hyperparams,
        entity="alessiochen98-university-of-florence",
        name=hyperparams["run_name"]
    )
    with wandb.init(**wandb_config):
        config = wandb.config
        barlowtwins(config)
