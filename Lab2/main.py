import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torchvision.models import resnet18

import wandb

from train import train_loop
from dataset import AugmentedImageDataset, MakeDataLoaders



def barlowtwins(config):
    from nets import SiameseNetSync as SiameseNet
    from barlow import BarlowTwins

    ### Dataset and DataLoader
    trainset = AugmentedImageDataset()
    # testset = CustomMNIST(train=False)

    loader = MakeDataLoaders(trainset, config.batch_size)
    train_loader = loader.loader
    # test_loader = _loader_train.test_dataloader
    # test_loader = DataLoader(
    #     testset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    ### Model and Loss function
    backbone = resnet18()
    model = SiameseNet(backbone)
    criterion = BarlowTwins(0.005)
    model = model.to(config.device)
    criterion = criterion.to(config.device)

    ### Optimizer
    params = [{
        "params": model.parameters(),  # backbone
        "params": criterion.projector.parameters(),  # projector
        "params": criterion.bn.parameters()  # batch norm layer
    }]
    optimizer = optim.Adam(
        params,
        lr=config.learning_rate,
        # weight_decay=config.weight_decay
    )

    ### Start training
    train_loop(model, train_loader, criterion, optimizer, config)
    ### Test model
    # test_acc = test(model, test_loader, config)
    # print(f"Test accuracy: {test_acc}")
    # wandb.log({"test_accuracy": test_acc})

    ### Save the model in the exchangeable ONNX format
    # input_data = torch.rand(1, 3, 32, 32).to(config.device)
    # torch.onnx.export(model, (input_data, input_data), "model.onnx")
    # wandb.save("model.onnx")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML COnfiguration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.SafeLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["device"] = device

    wandb_config = dict(
        project="comp-viz",
        config=hyperparams,
        entity="david-inf-team",
        name="barlow"
    )
    with wandb.init(**wandb_config):
        config = wandb.config
        barlowtwins(config)
