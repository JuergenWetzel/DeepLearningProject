from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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