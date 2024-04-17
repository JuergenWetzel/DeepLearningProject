import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
batch_size = 100

train_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform, split="trainval")
test_set = datasets.OxfordIIITPet(root="Dataset", download=False, transform=transform, split="test")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

print(len(train_loader))

model = resnet18()
for param in model.parameters():
   param.requires_grad = False
num_features = model.fc.in_features
new_classification_layer = torch.nn.Linear(num_features, 2)
model.fc = new_classification_layer
