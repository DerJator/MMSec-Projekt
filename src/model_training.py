import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import Casia2
import models
import first_compressions as cmp
import argparse
import wandb
import random

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=torch.device("cpu"), log_interval=1):
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        """ TRAINING LOOP """
        model.train()

        for img1, img2, label in train_loader:
            # print(img1[0].size(), img1[1].size(), label)
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            torch.cuda.empty_cache()

            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        """ VALIDATION LOOP """
        model.eval()
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)

                val_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        if epoch % log_interval == 0:
            run.log({"train-loss": epoch_train_loss, "val-loss": epoch_val_loss})  # Max 1x pro epoch (bei mehreren Sachen z.B. dict füllen)
        print(f'Epoch {epoch + 1}/{num_epochs} \t| Train Loss: {epoch_train_loss:.4f} \t| Val Loss: {epoch_val_loss:.4f}')


def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            output1, output2 = model(img1, img2)

            loss = loss_fun(output1, output2, label)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')

    # Add arguments
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--workdir', type=str, required=True, help='Working directory')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    DATA_PATH = args.dataset_path
    WORKDIR = args.workdir
    print(f"{DATA_PATH=}\n{WORKDIR=}")

    N_CHANNELS = 192
    DEVICE = torch.device("cuda")
    # DEVICE = torch.device("cpu")

    """ Initialize Dataset """
    casia_data = Casia2.Casia2Dataset(DATA_PATH, channels=[i for i in range(N_CHANNELS)])
    casia_data.organize_output_pairs(mode="neg_pos")  # Also add positive pairs
    # print(casia_data.output_pairs)
    train_pairs, test_pairs, train_labels, test_labels = casia_data.train_test_split(test=0.15)

    train_size = int(0.8235 * len(train_pairs))  # 0.8235 * 75% is 70% of the original data
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
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    """ Package in Dataset and DataLoader """
    train_data = models.ImagePairsDataset(train_pairs, train_labels, transform)
    val_data = models.ImagePairsDataset(val_pairs, val_labels, transform)
    test_data = models.ImagePairsDataset(test_pairs, test_labels, transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # for i1, i2, label in train_loader:
        # print(type(i1), type(i2), type(label))
        # break

    """ Put images through AI Coder """
    compressor = cmp.Compressor(cmp.CModelName.CHENG2020, device=DEVICE)
    train_cmp_loader = compressor.compress_torch(train_loader)
    val_cmp_loader = compressor.compress_torch(val_loader)
    test_cmp_loader = compressor.compress_torch(test_loader)

    """ FIRST PART: Train all Au-Tp pairs and the same number of Au-Au / Tp-Tp pairs """

    model = models.SiameseNetwork(in_channels=N_CHANNELS).to(DEVICE)
    loss_fun = models.ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # init wandb
    run = wandb.init(project="Triplet1")
    # save model inputs and model parameters
    config = run.config
    config.dropout = 0.01
    # Log gradients
    run.watch(model)

    """ TRAINING """

    train_model(model, train_cmp_loader, val_cmp_loader, loss_fun, optimizer, device=DEVICE, num_epochs=50)

    # Evaluate the model
    evaluate_model(model, test_cmp_loader)

