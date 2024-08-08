import os.path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import Casia2
import models
import losses
import first_compressions as cmp
import helpers

import argparse
import wandb
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def calc_loss(loss_func, z_au, z_tp, label_pair: tuple = None, nn_model=None, device=torch.device('cuda')):
    label1, label2 = label_pair

    if isinstance(loss_func, losses.ContrastiveLoss):
        diff_label = torch.Tensor([(0 if label1[i] == label2[i] else 1) for i in range(len(label1))]).to(device)
        loss = loss_func(z_au.to(device), z_tp.to(device), diff_label)
        # print(f"Contrastive {loss=}")

    elif isinstance(loss_func, losses.ESupConLoss):
        prototypes = nn_model.get_weights_class_layer()
        loss = loss_func(z_au.to(device), z_tp.to(device), prototypes.to(device))
        del prototypes
        # print(f"ESupCon {loss=}")
    else:
        raise ValueError('Unsupported criterion')

    del label1, label2

    return loss


def train_model(model, train_loader, val_loader, criterion, optimizer, model_version, num_epochs=10, device=torch.device("cpu"), log_interval=1):

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
            # torch.cuda.empty_cache()

            z_au, y_au, z_tp, y_tp = model(img1, img2)

            loss = calc_loss(criterion, z_au, z_tp, label_pair=(label1, label2), nn_model=model)
            loss.backward()
            optimizer.step()

            del z_au, y_au, z_tp, y_tp, img1, img2

            train_loss += float(loss)  # Tensor with grad (memory!) -> float

        """ VALIDATION LOOP """
        model.eval()
        best_val_loss = float("inf")
        with torch.no_grad():
            for img1, img2, label in val_loader:
                label1, label2 = label[0].to(device), label[1].to(device)
                img1, img2 = img1.to(device), img2.to(device)

                z_au, y_au, z_tp, y_tp = model(img1, img2)
                loss = calc_loss(criterion, z_au, z_tp, label_pair=(label1, label2), nn_model=model)
                del z_au, y_au, z_tp, y_tp, img1, img2

                val_loss += float(loss)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        if epoch % log_interval == 0:
            run.log({"train-loss": epoch_train_loss, "val-loss": epoch_val_loss})  # Max 1x pro epoch (bei mehreren Sachen z.B. dict füllen)
        print(f'Epoch {epoch + 1}/{num_epochs} \t| Train Loss: {epoch_train_loss:.4f} \t| Val Loss: {epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(f"Saving model at epoch {epoch + 1}")
            torch.save(model.state_dict(), f'../models/model_{epoch + 1}.pth')

    print("Training is completed!")
    print(f"Saving last model at epoch {epoch + 1}")
    torch.save(model.state_dict(), f'./models/model{model_version}_ep{epoch + 1}.pth')

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
    all_labels = []

    for data_loader in data_loaders:
        for img1, img2, labels in data_loader:  # img: tensor, labels: list with 2 tensors, 1 for each img tensor
            # print(f"{type(labels)=}")
            # print(f"{labels=}")  # Labels is list of two tensors with batchsize
            # print(f"{label1=}, {label2=}")
            for img in (img1, img2):
                img_repr = model.forward_representation(img).cpu().detach()
                representations.append(img_repr)

            repr_tensor = torch.cat(representations, dim=0)
            # print(f"{repr_tensor.size()=}")

            for label_tensor in labels:
                all_labels.extend(label_tensor.cpu().tolist())

    # representations = np.concatenate(representations, axis=0)
    if not as_torch:
        repr_tensor = repr_tensor.numpy()
        all_labels = np.array(all_labels)
    else:
        all_labels = torch.Tensor(all_labels)

    print(f"{repr_tensor.shape=}")
    print(f"{all_labels.shape=}")

    return repr_tensor, all_labels


def print_memory_summary():
    print(torch.cuda.memory_summary())


def svm_classification(train_samples, train_classes, test_samples, test_classes):

    # Create an SVM classifier pipeline with standard scaling
    svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    svm_classifier.fit(train_samples, train_classes)

    # Predict on the test set
    test_preds = svm_classifier.predict(test_samples)

    # Calculate the accuracy
    accuracy = accuracy_score(test_classes, test_preds)
    print(f"SVM Accuracy: {accuracy * 100:.2f}%")
    wandb.log({"SVM Accuracy": accuracy})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')

    # Add arguments
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--workdir', type=str, required=True, help='Working directory')
    parser.add_argument('--version', type=int, default=2, help='Version of the model (atm 1 or 2)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--channel_selection', type=bool, default=True, help='Whether to use channel selection')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    DATA_PATH = args.dataset_path
    WORKDIR = args.workdir
    MODEL_VERSION = args.version
    print(f"{DATA_PATH=}\n{WORKDIR=}\n{MODEL_VERSION=}")

    N_EPOCHS = args.epochs
    DEVICE = torch.device("cuda")
    BATCH_SIZE = args.batch_size
    if args.channel_selection:
        SELECTED_CHANNELS = [4, 29, 37, 39, 47, 49, 52, 55, 65, 85, 100, 101, 102, 110, 125, 131,  151, 179, 185]
        N_CHANNELS = len(SELECTED_CHANNELS)
    else:
        N_CHANNELS = 192
        SELECTED_CHANNELS = [i for i in range(N_CHANNELS)]

    # DEVICE = torch.device("cpu")

    """ Initialize Dataset """
    train_loader, val_loader, test_loader = helpers.init_dataset(DATA_PATH, SELECTED_CHANNELS, MODEL_VERSION, BATCH_SIZE)

    """ Put images through AI Coder """
    compressor = cmp.Compressor(cmp.CModelName.CHENG2020, device=DEVICE, channels=SELECTED_CHANNELS)
    train_cmp_loader = compressor.compress_torch(train_loader)
    val_cmp_loader = compressor.compress_torch(val_loader)
    test_cmp_loader = compressor.compress_torch(test_loader)

    """ FIRST PART: Train all Au-Tp pairs and the same number of Au-Au / Tp-Tp pairs """

    model_name = f'./models/model{MODEL_VERSION}_ch{N_CHANNELS}_ep{N_EPOCHS}.pth'
    if MODEL_VERSION == 1:
        model = models.SiameseNetwork(in_channels=N_CHANNELS, class_layer=False).to(DEVICE)
        loss_fun = losses.ContrastiveLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif MODEL_VERSION == 2:
        model = models.SiameseNetwork(in_channels=N_CHANNELS, class_layer=True).to(DEVICE)
        loss_fun = losses.ESupConLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError('Unsupported model version')

    # init wandb
    proj_name = "Contrastive" if MODEL_VERSION == 1 else "ESupCon"
    run = wandb.init(project=proj_name)
    # save model inputs and model parameters
    config = run.config
    # Log gradients
    run.watch(model)

    """ TRAINING """
    if not os.path.exists(model_name):
        trained_model = train_model(model, train_cmp_loader, val_cmp_loader, loss_fun, optimizer,
                                    model_version=MODEL_VERSION, device=DEVICE, num_epochs=N_EPOCHS)
    else:
        print("Found existing model, take that")
        trained_model = models.SiameseNetwork(in_channels=N_CHANNELS, class_layer=(MODEL_VERSION == 2)).to(DEVICE)
        trained_model.load_state_dict(torch.load(model_name))

    """ EVALUATION """
    torch.cuda.empty_cache()
    test_representations, test_labels = get_representations(trained_model, [test_cmp_loader], as_torch=False)  #!
    print(f"{type(test_representations)=}")
    print(f"{type(test_labels)=}")
    print(test_representations[0])
    print(test_labels)
    helpers.tensorboard_vis(test_representations, test_labels, N_EPOCHS)
    print(f"{len(test_representations)=}")
    print(type(test_representations))

    if MODEL_VERSION == 1:
        trained_model.eval()
        train_representations, train_class = get_representations(trained_model, [train_cmp_loader, val_cmp_loader], as_torch=False)
        svm_classification(train_representations, train_class, test_representations, test_labels)

    elif MODEL_VERSION == 2:
        trained_model.eval()

        test_labels = []
        predictions = []
        for img1, img2, label_pairs in test_cmp_loader:
            for img in [img1, img2]:
                probs = trained_model.classify_sample(img.to(DEVICE)).cpu()
                label = torch.argmax(probs, dim=1)
                predictions.extend(label.tolist())
            test_labels.extend(label_pairs[0].cpu().tolist())
            test_labels.extend(label_pairs[1].cpu().tolist())

        # Calculate the accuracy
        print(predictions)
        print(test_labels)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"ESupCon Accuracy: {accuracy * 100:.2f}%")
        wandb.log({"ESupCon Accuracy": accuracy})

    # Evaluate the model
    # evaluate_model(model, test_cmp_loader)

