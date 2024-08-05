import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
        print(f"input1: {input1.size()}")
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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, pos_factor=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_factor = pos_factor

    def forward(self, output1, output2, diff_label):
        """
            diff_label: 1 if au-tp sample 0 if samples from same class
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(-diff_label * torch.pow(euclidean_distance, 2) +
                                      self.pos_factor * (1 - diff_label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class ESupConLoss(nn.Module):
    def __init__(self, alpha=2):
        """
        alpha: weighting term of Au-Tp repulsion from same image
        """
        self.alpha = alpha
        super(ESupConLoss, self).__init__()

    def forward(self, z_au, z_tp, fc_weights: torch.Tensor):
        # outputs: Tensor of shape (batch_size, num_classes)
        # labels: Tensor of shape (batch_size,)
        self.n = z_au.size(0)
        self.n_k = self.n  # In each sample pair, there is 1 of each class
        print(f"{fc_weights.size()}")
        pt = fc_weights[0], fc_weights[1]  # TODO: Normalization

        supcon_loss = 0
        for i in range(self.n):
            supcon_loss += (-self.pos_loss(z_au, z_tp, i)
                            + self.neg_loss(z_au, z_tp, i)
                            + self.alpha * self.cosim(z_au[i], z_tp[i]))

        esupcon_loss = (1 / (self.n + 2) *
                        (self.pt_loss(z_au, z_tp, pt)
                         + supcon_loss))

        # print(f"{esupcon_loss.size()=}")  # TODO: Needs to be scalar, is a Tensor atm
        return torch.sum(esupcon_loss)

    def pt_loss(self, z_au: torch.Tensor, z_tp: torch.Tensor, pts: tuple):
        """
        Calculates loss of samples to prototypes. Rewards closeness to prototype of same class, punishes closeness to
        prototype of other class.

        Parameters:
        - z_au: Representations of class 0
        - z_tp: Representations of class 1
        - pts: Prototypes for each class

        Returns:
            Prototype Loss
        """
        loss = 0
        samples = (z_au, z_tp)
        for i in range(self.n):  # O(K*N)
            # pull prototype of same class, push prototype of other class
            loss += -self.cosim(samples[0][i], pts[0]) + self.cosim(samples[0][i], pts[1])
            loss += -self.cosim(samples[1][i], pts[1]) + self.cosim(samples[1][i], pts[0])

        loss *= 1 / self.n_k

        return loss

    def pos_loss(self, z_au, z_tp, ix, tau=1):
        """
        Calculates loss of positives for sample z_ix. Positives are other representations of the same class.

        Returns: Positive Loss
        """
        loss = torch.zeros_like(z_au)
        for j in range(self.n):
           if j != ix:
               loss += torch.exp(self.cosim(z_au[ix], z_au[j]) / tau)
               loss += torch.exp(self.cosim(z_tp[ix], z_tp[j]) / tau)

        loss = torch.log(loss)

        return loss

    def neg_loss(self, z_au, z_tp, ix, tau=1):
        """
        Calculates loss of negatives for sample z_ix. Negatives are other representations of the other class.

        Returns: Negative Loss
        """
        loss = 0
        for j in range(self.n):
            if j != ix:
                loss += torch.exp(self.cosim(z_au[ix], z_tp[j]) / tau)

        loss = torch.log(loss)

        return loss

    def cosim(self, x1, x2):
        return torch.matmul(x1, x2)


class ImagePairsDataset(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
