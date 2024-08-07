import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, in_channels: int, class_layer: bool):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.class_layer = None
        if class_layer:
            self.class_layer = nn.Linear(64, 2)
            self.softmax = nn.Softmax(dim=1)

    def forward_representation(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = F.relu(self.fc3(x))

        return z

    def forward_classifier(self, z):
        return self.softmax(self.class_layer(z))

    def forward(self, input1, input2):
        y1, y2 = None, None
        # print(f"input1: {input1.size()}")
        z1 = self.forward_representation(input1)
        if self.class_layer:
            y1 = self.forward_classifier(z1)
        z2 = self.forward_representation(input2)
        if self.class_layer:
            y2 = self.forward_classifier(z2)

        return z1, y1, z2, y2

    def classify_sample(self, img):
        return self.class_layer(self.forward_representation(img))

    def get_weights_class_layer(self):
        return self.class_layer.weight
