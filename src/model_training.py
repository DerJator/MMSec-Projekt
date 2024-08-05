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
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


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

            z_au, y_au, z_tp, y_tp = model(img1, img2)
            if isinstance(criterion, models.ContrastiveLoss):
                diff_label = 0 if label[0] == label[1] else 1
                loss = criterion(z_au, z_tp, diff_label)
            elif isinstance(criterion, models.ESupConLoss):
                prototypes = model.get_weights_class_layer()
                loss = criterion(z_au, z_tp, prototypes.cpu())
            else:
                raise ValueError('Unsupported criterion')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        """ VALIDATION LOOP """
        best_val_loss = float("inf")
        model.eval()
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                z_au, z_tp = model(img1, img2)
                loss = criterion(z_au, z_tp, label)

                val_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        if epoch % log_interval == 0:
            run.log({"train-loss": epoch_train_loss, "val-loss": epoch_val_loss})  # Max 1x pro epoch (bei mehreren Sachen z.B. dict füllen)
        print(f'Epoch {epoch + 1}/{num_epochs} \t| Train Loss: {epoch_train_loss:.4f} \t| Val Loss: {epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(f"Saving model at epoch {epoch + 1}")
            torch.save(model.state_dict(), f'../models/model_{epoch}.pth')


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=torch.device("cpu"), log_interval=1):

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        """ TRAINING LOOP """
        model.train()

        for img1, img2, label in train_loader:
            label1, label2 = label[0].to(device), label[1].to(device)
            # print(img1[0].size(), img1[1].size(), label)
            img1, img2 = img1.to(device), img2.to(device)

            optimizer.zero_grad()
            torch.cuda.empty_cache()

            z_au, y_au, z_tp, y_tp = model(img1, img2)

            if isinstance(criterion, models.ContrastiveLoss):
                diff_label = torch.Tensor([(0 if label1[i] == label2[i] else 1) for i in range(len(label1))]).to(device)
                loss = criterion(z_au.to(device), z_tp.to(device), diff_label)
                print("Contrastive")
                print(loss)
            elif isinstance(criterion, models.ESupConLoss):
                prototypes = model.get_weights_class_layer()
                loss = criterion(z_au.to(device), z_tp.to(device), prototypes.cuda())
                del z_au, y_au, z_tp, y_tp, img1, img2, label1, label2, prototypes
                print("ESupCon")
                print(loss)
            else:
                raise ValueError('Unsupported criterion')
            loss.backward()
            optimizer.step()

            train_loss += float(loss)

        """ VALIDATION LOOP """
        model.eval()
        with torch.no_grad():
            for img1, img2, label in val_loader:
                label1, label2 = label[0].to(device), label[1].to(device)

                img1, img2 = img1.to(device), img2.to(device)
                z_au, y_au, z_tp, y_tp = model(img1, img2)
                if isinstance(criterion, models.ContrastiveLoss):
                    diff_label = torch.Tensor([(0 if label1[i] == label2[i] else 1) for i in range(len(label1))]).to(
                        device)
                    loss = criterion(z_au.to(device), z_tp.to(device), diff_label)
                elif isinstance(criterion, models.ESupConLoss):
                    prototypes = model.get_weights_class_layer()
                    loss = criterion(z_au.to(device), z_tp.to(device), prototypes.cuda())
                else:
                    raise ValueError('Unsupported criterion')

                val_loss += float(loss)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        if epoch % log_interval == 0:
            run.log({"train-loss": epoch_train_loss, "val-loss": epoch_val_loss})  # Max 1x pro epoch (bei mehreren Sachen z.B. dict füllen)
        print(f'Epoch {epoch + 1}/{num_epochs} \t| Train Loss: {epoch_train_loss:.4f} \t| Val Loss: {epoch_val_loss:.4f}')

    print("Training is completed!")
    return model


def evaluate_model(model, test_loader):
    pass
    # model.eval()
    # total_loss = 0.0
    # with torch.no_grad():
    #     for img1, img2, label in test_loader:
    #         img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
    #         output1, output2 = model(img1, img2)
    #
    #         loss = loss_fun(output1, output2, label)
    #         total_loss += loss.item()
    #
    # avg_loss = total_loss / len(test_loader)
    # print(f'Test Loss: {avg_loss:.4f}')


def get_representations(model, data_loaders, as_torch=True):
    # Collect representations of data (including validation data)
    representations = []
    labels = []

    for data_loader in data_loaders:
        for img1, img2, labels in data_loader:
            print(f"{labels=}")
            label1, label2 = labels[0].cpu(), labels[1].cpu()
            print(f"{label1=}, {label2=}")
            if as_torch:
                representations.append(model.forward_representation(img1).cpu().detach())
                representations.append(model.forward_representation(img2).cpu().detach())
            else:
                representations.append(model.forward_representation(img1).cpu().detach().numpy())
                representations.append(model.forward_representation(img2).cpu().detach().numpy())

            labels.extend([label1, label2])

    # representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels, axis=0)

    return representations, labels


def print_memory_summary():
    print(torch.cuda.memory_summary())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')

    # Add arguments
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--workdir', type=str, required=True, help='Working directory')
    parser.add_argument('--version', type=int, default=2, help='Version of the model (atm 1 or 2)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    DATA_PATH = args.dataset_path
    WORKDIR = args.workdir
    MODEL_VERSION = args.version
    print(f"{DATA_PATH=}\n{WORKDIR=}\n{MODEL_VERSION}")

    N_CHANNELS = 192
    N_EPOCHS = args.epochs
    DEVICE = torch.device("cuda")
    BATCH_SIZE = args.batch_size
    # DEVICE = torch.device("cpu")

    """ Initialize Dataset """
    casia_data = Casia2.Casia2Dataset(DATA_PATH, channels=[i for i in range(N_CHANNELS)])
    if MODEL_VERSION == 1:
        casia_data.organize_output_pairs(mode="neg_pos")  # Also add positive pairs for contrastive loss
    elif MODEL_VERSION == 2:
        casia_data.organize_output_pairs(mode="neg")

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

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


    """ Put images through AI Coder """
    compressor = cmp.Compressor(cmp.CModelName.CHENG2020, device=DEVICE)
    train_cmp_loader = compressor.compress_torch(train_loader)
    val_cmp_loader = compressor.compress_torch(val_loader)
    test_cmp_loader = compressor.compress_torch(test_loader)

    """ FIRST PART: Train all Au-Tp pairs and the same number of Au-Au / Tp-Tp pairs """

    if MODEL_VERSION == 1:
        model = models.SiameseNetwork(in_channels=N_CHANNELS, class_layer=False).to(DEVICE)
        loss_fun = models.ContrastiveLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif MODEL_VERSION == 2:
        model = models.SiameseNetwork(in_channels=N_CHANNELS, class_layer=True).to(DEVICE)
        loss_fun = models.ESupConLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError('Unsupported model version')

    # init wandb
    proj_name = "Contrastive" if MODEL_VERSION == 1 else "ESupCon"
    run = wandb.init(project=proj_name)
    # save model inputs and model parameters
    config = run.config
    config.dropout = 0.01
    # Log gradients
    run.watch(model)

    """ TRAINING """
    trained_model = train_model(model, train_cmp_loader, val_cmp_loader, loss_fun, optimizer, device=DEVICE, num_epochs=N_EPOCHS)

    """ EVALUATION """
    torch.cuda.empty_cache()
    test_representations, test_labels = get_representations(trained_model, [test_cmp_loader])
    # print(test_representations)
    # print(type(test_representations))

    if MODEL_VERSION == 1:
        trained_model.eval()
        train_representations, train_labels = get_representations(trained_model, [train_cmp_loader, val_cmp_loader])

        print(f"train repr: {len(train_representations)}, train_labels: {len(train_labels)}")
        print(f"test repr: {len(test_representations)}, test_labels: {len(test_labels)}")
        # Create an SVM classifier pipeline with standard scaling
        svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        svm_classifier.fit(train_representations, train_labels)

        # Predict on the test set
        test_preds = svm_classifier.predict(test_representations)

        # Calculate the accuracy
        accuracy = accuracy_score(test_labels, test_preds)
        print(f"SVM Accuracy: {accuracy * 100:.2f}%")
        wandb.log({"SVM Accuracy": accuracy})

    elif MODEL_VERSION == 2:
        trained_model.eval()

        test_labels = []
        predictions = []
        for img1, img2, labels in test_cmp_loader:
            for img in [img1, img2]:
                probs = trained_model.classify_sample(img.to(DEVICE)).cpu()
                label = torch.argmax(probs, dim=1)
                predictions.extend(label.tolist())
            test_labels.extend(labels[0].cpu().tolist())
            test_labels.extend(labels[1].cpu().tolist())

        # Calculate the accuracy
        print(predictions)
        print(test_labels)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"ESupCon Accuracy: {accuracy * 100:.2f}%")
        wandb.log({"ESupCon Accuracy": accuracy})

    # Evaluate the model
    # evaluate_model(model, test_cmp_loader)

