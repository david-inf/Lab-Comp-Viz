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
    from torch.utils.data import DataLoader
    from nets import SiameseNetSync as SiameseNet
    from barlow import BarlowTwins

    ## **** Pre-training **** ##
    ### Dataset and DataLoader
    ssl_trainset = AugmentedImageDataset()
    ssl_train_loader = DataLoader(
        ssl_trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)

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
    ssl_optimizer = optim.Adam(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    ### Run pre-training
    epoch_ssl, loss_ssl = train_loop_ssl(
        model, ssl_train_loader, criterion, ssl_optimizer, config
    )
    print("Pre-training done!")

    ## **** Evaluation **** ##
    ### Linear probe training
    train_data = BaseDataset()
    testset = BaseDataset(train=False)
    loader = MakeDataLoaders(train_data, testset, config)
    train_loader = loader.train_loader
    val_loader = loader.val_loader
    test_loader = loader.test_loader

    ### Run supervised training
    epoch_eval, loss_eval, acc_eval = linear_probe(
        backbone, model.in_features, train_loader, config
    )
    print("Training linear layer done!")

    ### Test model
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
        "optimizer_state_dict": ssl_optimizer.state_dict()
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
