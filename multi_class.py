import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights, resnet34
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2 as v2
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import random
from util import *
import os

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "model_name": "resnet18",
    "binary_classification": True,
    "adamW": False,

    "augmentation": True,
    "batch_train": 300,
    "batch_test": 500,
    "test_size": 1000,

    "lr": 0.003,
    "lr_backbone": 0.0001,
    "weight_decay": 0.01,
    "lr_decay": 0.5,
    "epochs": 3,
    "lr_step": 2,

    "tune_layers": ["fc"],
    "unfreeze_layers": ["layer4"],
    "freeze_epoch": 2
}

train_loader, test_loader = get_loaders(params)

model = get_model(params["model_name"])

# show_grad_stat(model)

freeze(model, params["tune_layers"])

criterion = nn.CrossEntropyLoss()
if params["adamW"]:
    optimizer = torch.optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": params["lr"], "weight_decay": params["weight_decay"]},
            {"params": [param for name, param in model.named_parameters() if 'fc' not in name],
             "lr": params["lr_backbone"],
             "weight_decay": params["weight_decay"]},
        ]
    )

else:
    optimizer = torch.optim.Adam(
        [
            {"params": model.fc.parameters(), "lr": params["lr"], "weight_decay": params["weight_decay"]},
            {"params": [param for name, param in model.named_parameters() if 'fc' not in name],
             "lr": params["lr_backbone"],
             "weight_decay": params["weight_decay"]},
        ]
    )

scheduler = StepLR(optimizer, step_size=params["lr_step"], gamma=params["lr_decay"])

train_model(model, train_loader, criterion, optimizer, scheduler, params)
acc = evaluate_model(model, test_loader)

save_params(params, acc)
