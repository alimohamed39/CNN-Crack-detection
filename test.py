import torch
from model import get_model
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


model = get_model()



model.load_state_dict(torch.load(r'C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\Surface Crack detection proj\best_model.pth'))
model.eval()

transform = transforms.Compose([transforms.Resize((224,224))
                                   ,transforms.ToTensor()
                                   , transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


test_dataset = datasets.ImageFolder(root=r"C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\Surface Crack detection proj\test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
total , correct = 0, 0
with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)
        _,prediction = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
