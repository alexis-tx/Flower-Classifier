
import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        # series of convolutional layers 
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3 ,out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),

            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),

            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),
        )
        # series of fully connected layers 
        self.linear_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 102)
        )

    # this function gets the moddel to go through the layers above 
    def forward(self, x):
        x = self.conv_stack(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_stack(x)

        return x