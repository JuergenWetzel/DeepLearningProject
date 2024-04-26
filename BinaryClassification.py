import os
import random
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet34, ResNet34_Weights

# map breed -> cat or dog
class_mapping = [
    0,  # Class 0: Abyssinian - Cat
    1,  # Class 1: american Bulldog - Dog
    1,  # Class 2: american Pit Bull Terrier - Dog
    1,  # Class 3: basset Hound - Dog
    1,  # Class 4: beagle - Dog
    0,  # Class 5: Bengal - Cat
    0,  # Class 6: Birman - Cat
    0,  # Class 7: Bombay - Cat
    1,  # Class 8: boxer - Dog
    0,  # Class 9: British Shorthair - Cat
    1,  # Class 10: chihuahua - Dog
    0,  # Class 11: Egyptian Mau - Cat
    1,  # Class 12: english Cocker Spaniel - Dog
    1,  # Class 13: english Setter - Dog
    1,  # Class 14: german Shorthaired - Dog
    1,  # Class 15: great Pyrenees - Dog
    1,  # Class 16: havanese - Dog
    1,  # Class 17: japanese Chin - Dog
    1,  # Class 18: keeshond - Dog
    1,  # Class 19: leonberger - Cat
    0,  # Class 20: Maine Coon - Cat
    1,  # Class 21: miniature Pinscher - Dog
    1,  # Class 22: newfoundland - Dog
    0,  # Class 23: Persian - Cat
    1,  # Class 24: pomeranian - Dog
    1,  # Class 25: pug - Dog
    0,  # Class 26: Ragdoll - Cat
    0,  # Class 27: Russian Blue - Cat
    1,  # Class 28: saint Bernard - Dog
    1,  # Class 29: samoyed - Dog
    1,  # Class 30: scottish Terrier - Dog
    1,  # Class 31: shiba Inu - Dog
    0,  # Class 32: Siamese - Cat
    0,  # Class 33: Sphynx - Cat
    1,  # Class 34: staffordshire Bull Terrier - Dog
    1,  # Class 35: wheaten Terrier - Dog
    1,  # Class 36: yorkshire Terrier - Dog
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# maps the 37 classes to 0 cat or 1 dog
def binary_transform(target):
    return class_mapping[target]


# loads the training and test data.
# test data has an amount of test_size, training data are all other images
def load_data(test_size=1000, batch_train=100, batch_test=1000):
    train_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform, split="trainval",
                                       target_transform=binary_transform)
    test_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform, split="test",
                                      target_transform=binary_transform)
    dataset = ConcatDataset([train_set, test_set])
    shuffled = list(range(len(dataset)))
    random.shuffle(shuffled)
    train_set = Subset(dataset, shuffled[test_size:])
    test_set = Subset(dataset, shuffled[0:test_size])
    trainloader = DataLoader(train_set, batch_size=batch_train, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_test, shuffle=False)
    return trainloader, testloader


train_loader, test_loader = load_data(1500, 100, 1500)


# ResNet34 model with 2 outputs
# All layers except the new fc layer aren't calculating the gradient
class BinaryResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = resnet34(weights=ResNet34_Weights.DEFAULT)
        for param in self.network.parameters():
            param.requires_grad = False
        num_features = self.network.fc.in_features
        new_classification_layer = torch.nn.Linear(num_features, 2)
        self.network.fc = new_classification_layer

    def forward(self, x):
        return F.softmax(self.network(x), dim=1)


# path and filename to save the model and csv
path = "./"
csvname = "accuracy_bin_long.csv"
modelname = "model_w"
# If a new csv should be used, it creates the file empty and adds the header
with open(path + csvname, "w") as file:
    file.write("weight, accuracy, loss, epochs, cat accuracy, dog accuracy\n")


# calculates the total accuracy and the accuracy of each class
# If classes is None it calculates only the total accuracy
def accuracy(model, dataloader, classes=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    correct = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if classes is not None:
            accs = accuracy_per_class(predicted, labels, classes)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    if classes is None:
        return acc
    else:
        return acc, accs


# Calculates the accuracy for each class
def accuracy_per_class(predicted, labels, classes):
    totals = []
    corrects = []
    accs = []
    for _ in range(classes):
        totals.append(0)
        corrects.append(0)
    for n in range(len(labels)):
        totals[labels[n]] += 1
        corrects[labels[n]] += int(labels[n] == predicted[n])
    for n in range(len(totals)):
        if totals == 0:
            accs.append(-1)
        else:
            accs.append(corrects[n] / totals[n])
    return accs


# trains the model with given dataloader, learning rate and weight decay for epoch epochs
def train_model(model, dataloader, epochs=5, lr=0.001, weight=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight)
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('[%2d] loss: %.3f' % (epoch + 1, loss.item()))
    return loss.item()


def best_weight(weights, epochs=10, lr=0.001, csv_name="weights.csv", model_name="model_w=", save_models=False, save_best_model=True):
    best_weight = 0
    best_acc = 0
    for weight in weights:
        print("Weight: %.5f" % weight)
        model = BinaryResNet34()
        loss = train_model(model, train_loader, weight, lr, epochs)
        train_acc = 100 * accuracy(model, train_loader)
        print("accuracy training: %.2f %%" % train_acc)
        acc, accs = accuracy(model, test_loader, 2)
        acc *= 100
        accs = [100 * a for a in accs]
        print("accuracy test: %.2f %%" % acc)
        if save_models:
            torch.save(model.state_dict(), model_name + "%.5f.pt" % weight)
        elif save_best_model:
            if acc > best_acc:
                os.remove(model_name + "%.5f.pt" % best_weight)
                best_weight = weight
                best_acc = acc
                torch.save(model.state_dict(), model_name + "%.5f.pt" % best_weight)
        update_csv(weight, loss, epochs, train_acc, acc, accs, csv_name)


def update_csv(weight, loss, epochs, train_acc, test_acc, acc_per_class, filename):
    csv = "%.5f, %.4f, %d, %.3f, %.3f" % (weight, loss, epochs, train_acc, test_acc)
    for acc in acc_per_class:
        csv += ", %.3f" % acc
    csv += "\n"
    with open(filename, "a") as file:
        file.write(csv)


best_weight([0.5], 15, 0.001, "accuracy_bin_long.csv")
