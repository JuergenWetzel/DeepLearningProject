import random
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet34, ResNet34_Weights

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


def binary_transform(target):
    return class_mapping[target]


def load_data(test_size, batch_train, batch_test):
    train_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform, split="trainval", target_transform=binary_transform)
    test_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform, split="test", target_transform=binary_transform)
    dataset = ConcatDataset([train_set, test_set])
    shuffled = list(range(len(dataset)))
    random.shuffle(shuffled)
    train_set = Subset(dataset, shuffled[test_size:])
    test_set = Subset(dataset, shuffled[0:test_size])
    trainloader = DataLoader(train_set, batch_size=batch_train, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_test, shuffle=False)
    return trainloader, testloader


train_loader, test_loader = load_data(1500, 100, 1500)


class DCResNet18(nn.Module):
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


# path = "/content/drive/MyDrive/DeepLearning/"
path = "./"
filename = "accuracy_bin_long.csv"
with open(path + filename, "w") as file:
    file.write("weight, accuracy, loss, epochs, cat accuracy, dog accuracy\n")


def accuracy(dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    correct = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        totals = []
        corrects = []
        for n in range(2):
            totals.append(0)
            corrects.append(0)
        for n in range(len(labels)):
            totals[labels[n]] += 1
            corrects[labels[n]] += int(labels[n] == predicted[n])
        accs = []
        for n in range(len(totals)):
            if totals == 0:
                accs.append(-1)
            else:
                accs.append(corrects[n] / totals[n])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc, accs


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


for weight in [0.5]:
    print("Weight: %.2f" % weight)
    epochs = 15
    model = DCResNet18()
    loss = train_model(model, train_loader, weight=weight, epochs=epochs)
    torch.save(model.state_dict(), path + "model_w=%.5f.pt" % weight)
    print("accuracy training: " + str(100 * accuracy(train_loader)[0]) + "%")
    acc, accs = accuracy(test_loader)
    print("accuracy test: " + str(acc))
    
    result = "%.5f,%.3f,%.5f,%d" % (weight, 100 * acc, loss, epochs)
    for acc in accs:
        acc = 100 * acc
        result += ", %.2f" % acc
    with open(path + filename, "a") as file:
        file.write(result + "\n")
