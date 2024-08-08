from torch import min as torch_min, max as torch_max
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import os
import random
import Casia2


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


def tensorboard_vis(representations, labels, n_epochs):
    # Directory to save the TensorBoard logs
    log_dir = './tensorboard_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Initialize the TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir)

    # Add embeddings and metadata to TensorBoard
    writer.add_embedding(representations, metadata=labels, tag=f'Contrastive_v1_e{n_epochs}')

    # Close the writer
    writer.close()


def init_dataset(DATA_PATH, SELECTED_CHANNELS, MODEL_VERSION, BATCH_SIZE):
    casia_data = Casia2.Casia2Dataset(DATA_PATH, channels=SELECTED_CHANNELS)
    if MODEL_VERSION == 1:
        casia_data.organize_output_pairs(mode="neg_pos")  # Also add positive pairs for contrastive loss
    elif MODEL_VERSION == 2:
        casia_data.organize_output_pairs(mode="neg")

    # print(casia_data.output_pairs)
    train_pairs, test_pairs, train_labels, test_labels = casia_data.train_test_split(test=0.15)

    train_size = int(0.8235 * len(train_pairs))  # 0.8235 * 85% is 70% of the original data
    val_size = len(train_pairs) - train_size  # Remaining 15% of the original data

    all_indices = list(range(len(train_pairs)))
    random.shuffle(all_indices)
    train_ix = all_indices[:train_size]
    val_ix = all_indices[train_size:]

    val_pairs = [train_pairs[i] for i in val_ix]
    train_pairs = [train_pairs[i] for i in train_ix]
    val_labels = [train_labels[i] for i in val_ix]
    train_labels = [train_labels[i] for i in train_ix]
    print(f"TRAIN SIZE: {len(train_pairs)}, VAL SIZE: {len(val_pairs)}, TEST SIZE: {len(test_pairs)}")

    """ Transforms """
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    """ Package in Dataset and DataLoader """
    train_data = ImagePairsDataset(train_pairs, train_labels, transform)
    val_data = ImagePairsDataset(val_pairs, val_labels, transform)
    test_data = ImagePairsDataset(test_pairs, test_labels, transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader
