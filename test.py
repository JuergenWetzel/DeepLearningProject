import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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

batch_size = 100
def binary_transform(target):
    return class_mapping[target]


test_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform, target_transform=binary_transform, split="test")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

model = resnet18()
for param in model.parameters():
   param.requires_grad = False
num_features = model.fc.in_features
new_classification_layer = torch.nn.Linear(num_features, 2)
model.fc = new_classification_layer
model.load_state_dict(torch.load("model.pt"))
model.eval()

total = 0
correct = 0
for i, data in enumerate(test_loader):
    inputs, labels = data
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("accuracy: " + str(correct / total))
