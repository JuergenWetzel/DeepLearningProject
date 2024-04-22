import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights

print(torch.__version__)
print(torch.cuda.is_available())
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
batch_size = 100
class_mapping = [
    0,  # Class 0: Abyssinian - Cat
    1,  # Class 1: American Bulldog - Dog
    1,  # Class 2: American Pit Bull Terrier - Dog
    1,  # Class 3: Basset Hound - Dog
    1,  # Class 4: Beagle - Dog
    0,  # Class 5: Bengal - Cat
    0,  # Class 6: Birman - Cat
    0,  # Class 7: Bombay - Cat
    1,  # Class 8: Boxer - Dog
    0,  # Class 9: British Shorthair - Cat
    1,  # Class 10: Chihuahua - Dog
    0,  # Class 11: Egyptian Mau - Cat
    0,  # Class 12: English Cocker Spaniel - Cat
    0,  # Class 13: English Setter - Cat
    1,  # Class 14: German Shorthaired - Dog
    1,  # Class 15: Great Pyrenees - Dog
    0,  # Class 16: Havanese - Cat
    0,  # Class 17: Japanese Chin - Cat
    0,  # Class 18: Keeshond - Cat
    0,  # Class 19: Leonberger - Cat
    0,  # Class 20: Maine Coon - Cat
    1,  # Class 21: Miniature Pinscher - Dog
    1,  # Class 22: Newfoundland - Dog
    0,  # Class 23: Persian - Cat
    1,  # Class 24: Pomeranian - Dog
    1,  # Class 25: Pug - Dog
    0,  # Class 26: Ragdoll - Cat
    0,  # Class 27: Russian Blue - Cat
    1,  # Class 28: Saint Bernard - Dog
    1,  # Class 29: Samoyed - Dog
    0,  # Class 30: Scottish Terrier - Cat
    1,  # Class 31: Shiba Inu - Dog
    0,  # Class 32: Siamese - Cat
    0,  # Class 33: Sphynx - Cat
    1,  # Class 34: Staffordshire Bull Terrier - Dog
    1,  # Class 35: Wheaten Terrier - Dog
    1,  # Class 36: Yorkshire Terrier - Dog
]


def binary_transform(target):
    return class_mapping[target]


train_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform,
                                   target_transform=binary_transform, split="trainval")
test_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform,
                                  target_transform=binary_transform, split="test")
#
subset_size = 0.3
subset_lengths = [int(len(train_set) * subset_size), len(train_set) - int(len(train_set) * subset_size)]
train_set, subset_large = random_split(train_set, subset_lengths)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


#
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
# model = resnet18()
# for param in model.parameters():
#    param.requires_grad = False
# num_features = model.fc.in_features
# new_classification_layer = torch.nn.Linear(num_features, 2)
# model.fc = new_classification_layer
#
#
#
#
# # print(len(train_set))
# # print(train_set[0])
#
# # print(train_set[0][0].shape)
#
def train_model(model, train_set, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_set, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 3 == 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features
new_classification_layer = torch.nn.Linear(num_features, 2)
model.fc = new_classification_layer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print("Training model")
model.train()
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
print("Evaluating model")
model.eval()
evaluate_model(model, test_loader)
