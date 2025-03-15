# importing the relevant packages 
import torch
import torchvision
import torchvision.transforms.v2 as transforms 
from torch.utils.data import DataLoader
import torch.nn as nn
from model import NN

# data augmentation section using v2 transformations for training data
training_trans = transforms.Compose([
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Resize(116),
    transforms.RandomCrop((96,96)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ElasticTransform(),
    transforms.RandomPosterize(2),
    transforms.RandomAdjustSharpness(2),
    transforms.RandAugment(),
    transforms.Normalize(mean=[0.4701, 0.3985, 0.3177], std=[0.2577, 0.2059, 0.2193])
])

# data augmentation section using v2 transformations for testing and validation data 
test_trans = transforms.Compose([
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Resize(96),
    transforms.CenterCrop((96,96)),
    transforms.Normalize(mean=[0.4701, 0.3985, 0.3177], std=[0.2577, 0.2059, 0.2193])
])

# downloading the data from the dataset and applying the transformations mentioned above 
training_data = torchvision.datasets.Flowers102(
    root='data',
    split='train',
    transform=training_trans,
    download=True,
)


test_data = torchvision.datasets.Flowers102(
    root='data',
    split='test',
    transform=test_trans,
    download=True,
)


val_data = torchvision.datasets.Flowers102(
    root='data',
    split='val',
    transform=test_trans,
    download=True,
)

# initialising the batch size 
batch_size = 8

# loading in the data from the dataset using the augmented data, shuffling the images 
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


 # initialises the model   
model = NN()

# checking if the gpu is available to use 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
# moving the model to run on the device availbale
model.to(device)


# defining a loss function
criteria = nn.CrossEntropyLoss()
# defining an optimizer
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)

# the training function
def train(dataloader, model, criteria, optimiser):
    # training the model with the training data
    model.train()
    size = len(dataloader.dataset)
    # defining running loss function 
    loss = 0

    for batch, data in enumerate(dataloader):

        images, labels = data
        # moving the images and labels to the correct device 
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = criteria(pred, labels)

        # zeroes gradients
        optimiser.zero_grad()
        # computes the gradient 
        loss.backward()
        # paraneter update based on the latest gradient 
        optimiser.step()

        #calculating and outputting the average loss per batch size 
        if batch % 80 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# validation function
def validate(valloader, model, criteria):

    size = len(valloader.dataset)
    num_batches = len(valloader)
    # evaluating with the validation data
    model.eval()

    # defining the running test loss and the number of classes the model got right 
    test_loss, correct = 0, 0

    # not going to calculate the gradient since its not necessary in this stage of the program 
    with torch.no_grad():

        for data in valloader:
            images, labels = data
            # moving the images and labels to the correct device on the computer 
            images, labels = images.to(device), labels.to(device)

            # predicting what class the image will go into using the model
            pred = model(images)

            # adding to the running test loss and the number of images that have been filed to the correct class 
            test_loss += criteria(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    # calculating and outputting the average loss and validation accuracy 
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# define hyperparameter epochs 
epochs = 300
for k in range(epochs):
    print(f"Epoch {k+1}\n ------------------")
    # call the train function using the defined parameters
    train(train_dataloader, model, criteria, optimiser)
    # call the validation function using the defined parameters
    validate(val_dataloader, model, criteria)

# 
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

# outputs the accuracy percentage after using the trained model on testing data
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# save the model
torch.save(model.state_dict(), 'model.pth')
