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
    "batch_train": 200,
    "batch_test": 500,
    "test_size": 500,
    "lr": 0.001,
    "lr_backbone": 0.0001,
    "weight_decay": 0.001,
    "lr_decay": 0.5,
    "epochs": 2,
    "lr_step": 1,
    "tune_layers": ["fc"]
}


train_loader, test_loader = get_loaders(params)

model = get_model()

freeze(model, params["tune_layers"])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [
        {"params": model.fc.parameters(), "lr": params["lr"], "weight_decay": params["weight_decay"]},
        {"params": [param for name, param in model.named_parameters() if 'fc' not in name], "lr": params["lr_backbone"],
         "weight_decay": params["weight_decay"]},
    ]
)

scheduekr_backbone = StepLR(optimizer, step_size=params["lr_step"], gamma=params["lr_decay"])
scheduler_fc = StepLR(optimizer, step_size=params["lr_step"], gamma=params["lr_decay"])

train_model(model, train_loader, criterion, optimizer, scheduler_fc, scheduekr_backbone, params["epochs"])
acc = evaluate_model(model, test_loader)

save_params(params, acc)

