from torch import min as torch_min, max as torch_max
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


def min_max_scale(img_pair: tuple):
    norm_pair = []
    for i, img in enumerate(img_pair):
        min_val, max_val = torch_min(img), torch_max(img)
        norm_pair.append((img - min_val) / (max_val - min_val))

    return tuple(norm_pair)


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
