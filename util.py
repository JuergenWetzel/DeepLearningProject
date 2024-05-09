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
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_grad_stat(model):
    for name, param in model.named_parameters():
        print(name, param.requires_grad)


def get_loaders(params):
    test_size = params["test_size"]
    batch_train = params["batch_train"]
    batch_test = params["batch_test"]
    augmentation = params["augmentation"]

    transformRotations = transforms.Compose([
        transforms.Resize((460, 460)),
        v2.RandomHorizontalFlip(p=0.4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=(-10, 10)),
        # Apply affine transformation here
        # lambda x: F.affine(x, angle=0, translate=(0, 0), scale=1.0, shear=0),
        transforms.RandomResizedCrop(224, scale=(1, 1.0), ratio=(1, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if augmentation:
        train_set = datasets.OxfordIIITPet(root="Dataset", download=True, transform=transformRotations, split="trainval")
    else:
        train_set = datasets.OxfordIIITPet(root="Dataset", download=True, transform=transform, split="trainval")
    test_set = datasets.OxfordIIITPet(root="Dataset", download=True, transform=transform, split="test")

    dataset = ConcatDataset([train_set, test_set])
    shuffled = list(range(len(dataset)))
    random.shuffle(shuffled)
    train_set = Subset(dataset, shuffled[test_size:])
    test_set = Subset(dataset, shuffled[0:test_size])
    train_loader = DataLoader(train_set, batch_size=batch_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_test, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_set, criterion, optimizer, lr_scheduler_fc=None, lr_scheduler_back=None, num_epochs=10):
    losses = []
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        if epoch == 2:
            unfreeze(model)

        for i, data in enumerate(train_set, 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        lr_scheduler_fc.step()
        lr_scheduler_back.step()
        losses.append(running_loss)
        print("epoch ", epoch, "loss ", running_loss)
    plot_array(np.array(losses))


def evaluate_model(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}')
    return accuracy


def plot_array(arr):
    plt.plot(arr)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss in epochs')
    plt.grid(True)
    plt.show()


def unfreeze(model):
    for name, param in model.named_parameters():
        # if "bn" in name:
        #   continue
        param.requires_grad = True


def freeze(model, tune_layers):
    for name, param in model.named_parameters():
        if any(layer in name for layer in tune_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_model():
    model = resnet34()
    if os.path.exists("resnet34.pth"):
        model.load_state_dict(torch.load("resnet34.pth"))
    else:
        model = resnet34(pretrained=True)
        torch.save(model.state_dict(), "resnet34.pth")

    model.fc = nn.Linear(model.fc.in_features, 37)
    model = model.to(device)
    return model


def save_params(params, accuracy):
    with open("results.txt", "a") as f:
        f.write(str(accuracy) + "\t:" + str(params) + "\n")
