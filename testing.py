## need to load in 
import torch
import torchvision
import torchvision.transforms.v2 as transforms 
from torch.utils.data import DataLoader
import torch.nn as nn
from model import NN

test_trans = transforms.Compose([
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Resize(96),
    transforms.CenterCrop((96,96)),
    transforms.Normalize(mean=[0.4701, 0.3985, 0.3177], std=[0.2577, 0.2059, 0.2193])
])

test_data = torchvision.datasets.Flowers102(
    root='data',
    split='test',
    transform=test_trans,
    download=True,
)

batch_size = 8

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = NN()
# moving the model to run on the device availbale
model.to(device)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')