import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.OxfordIIITPet(root="Dataset", download=True, transform=transform)
