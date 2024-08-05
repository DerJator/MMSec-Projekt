import compressai as cai
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from compressai.ops import compute_padding
from compressai.models import CompressionModel
from torch.utils.data import DataLoader, Dataset
import os
import regex as re
from enum import Enum


class CModelName(Enum):
    CHENG2020 = 0


class CompressedDataset(Dataset):
    def __init__(self, dataset, transform, quantized=False):
        self.dataset = dataset
        self.transform = transform
        self.quantized = quantized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, img2, label = self.dataset[idx]
        comp1 = self.transform(img1)
        comp2 = self.transform(img2)

        #
        if not self.quantized:
            return comp1[0], comp2[0], label
        else:
            return comp1[1], comp2[1], label

class Compressor:
    def __init__(self, model_name: CModelName, device: torch.device):
        if model_name == CModelName.CHENG2020:
            self.model = cheng2020_model()
        else:
            raise RuntimeError(f"Unknown model name: {model_name}")

        self.device = device
        if device == torch.device("cpu"):
            self.detach = True
        else:
            self.detach = False

    def compress_single(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return compress_latent(img, self.model, self.device, self.detach)

    def compress_torch(self, data: DataLoader) -> DataLoader:
        transform = lambda img: self.compress_latent(img, self.model, self.device, detach=False)
        transformed_dataset = CompressedDataset(data.dataset, transform)
        transformed_dataloader = DataLoader(transformed_dataset, batch_size=data.batch_size, shuffle=False)
        return transformed_dataloader

    def compress_latent(self, img: np.ndarray | torch.Tensor, model: CompressionModel, device: torch.device, detach: bool = True):
        """
        Takes image and returns its latent representation and quantized latent representation
        """
        with torch.no_grad():
            if type(img) is np.ndarray:
                img = img_to_torch(img)
            elif type(img) is torch.Tensor and len(img.size()) == 3:
                img = img.unsqueeze(0)

            h, w = img.shape[-2:]
            pad, unpad = compute_padding(in_h=h, in_w=w, min_div=2 ** 6)

            padded_img = F.pad(input=img, pad=pad, mode="constant", value=0)

            padded_img = padded_img.to(device)
            model = model.to(device)

            y = model.g_a(padded_img)  # latent space direkt nach Encoder
            y_hat = model.gaussian_conditional.quantize(y, mode="dequantize")  # latent space nach Quantisierung

            if detach:
                y_hat = y_hat.squeeze(0).detach().cpu().numpy().copy()
            else:
                y = y.squeeze(0)
                y_hat = y_hat.squeeze(0)

            return y, y_hat


def cheng2020_model(quality: int = 6):
    model = cai.zoo.cheng2020_anchor(quality, pretrained=True)
    model.update()

    return model


def compress_latent(img: np.ndarray, model: CompressionModel, device: torch.device, detach: bool = True):
    """
    Takes image and returns its latent representation and quantized latent representation
    """
    img = img_to_torch(img)

    h, w = img.shape[-2:]
    pad, unpad = compute_padding(in_h=h, in_w=w, min_div=2 ** 6)

    padded_img = F.pad(input=img, pad=pad, mode="constant", value=0)

    padded_img = padded_img.to(device)
    model = model.to(device)

    y = model.g_a(padded_img)  # latent space direkt nach Encoder
    y_hat = model.gaussian_conditional.quantize(y, mode="dequantize")  # latent space nach Quantisierung

    if detach:
        y_hat = y_hat.squeeze(0).detach().cpu().numpy().copy()

    return y, y_hat


def img_to_torch(img: np.ndarray) -> torch.Tensor:
    img = torch.from_numpy(img.copy()).float()
    img = img.permute((2, 0, 1))
    img = img.unsqueeze(0)

    return img


def versus_plot(img1: np.ndarray, img2: np.ndarray, main_title: str,
                title1: str = "Original", title2: str = "Manipulated"):
    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title(main_title)

    # Plotting the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.imshow(img1, cmap='gray')  # Adjust the colormap if necessary
    plt.title(title1)

    # Plotting the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.imshow(img2, cmap='gray')  # Adjust the colormap if necessary
    plt.title(title2)

    plt.show()


def assess_mse(error_tensor: np.ndarray, name: str = ""):
    """
    Plots mean error over all channels of the image and MSE per channel as a bar plot
    """
    # Mean latent representation MSE over all channels
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Disable the x and y scales
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title(f"MSE original vs. manipulated - Mean over latent channels\n{name}")
    plt.imshow(np.mean(error_tensor, axis=0), cmap="plasma")

    # Collect the MSE of each latent channel (per channel, scalar)
    mean_mse_channel = np.mean(error_tensor, axis=(1, 2))
    plt.subplot(1, 2, 2)
    plt.title(f"MSE per channel\n{name}, argmax: {np.argmax(mean_mse_channel)}")
    plt.bar(np.arange(mean_mse_channel.shape[0]), mean_mse_channel)
    plt.show()


def assess_manipulated_images(img_pairs: list[tuple],
                              model: CompressionModel, plot_dims: list[int] = None, n_assess_dims: int = 20,
                              quantize: bool = False, device: torch.device = "cpu"):
    """
    For every pair of original and manipulated image, compress using given model and compare
    MSE between channels of compression. For every image, collect the 20 channel indices of max MSE and return
    this result as an array shape(n_imgs, n_assess_dims)
    """
    most_changed_channels = np.zeros((len(img_pairs), n_assess_dims))
    mse_tensor = np.zeros((n_assess_dims, len(img_pairs)))

    for i, (original_path, manip_path) in enumerate(img_pairs):
        test_img = plt.imread(manip_path)
        original_img = plt.imread(original_path)

        # Flip img if necessary
        if test_img.shape[0] == original_img.shape[1] and test_img.shape[1] == original_img.shape[0]:
            test_img = test_img.transpose(1, 0, 2)

        if test_img.shape != original_img.shape:
            print(manip_path, original_path)
            mse_tensor[:, i] = None
            continue

        test = test_img / np.max(test_img)
        original = original_img / np.max(original_img)

        file = re.search(r"/([^/]+)$", original_path).group(1)

        y_f, y_hat_f = compress_latent(test, model, device=device)
        y, y_hat = compress_latent(original, model, device=device)

        # Calculate MSE between latent channels
        if quantize:
            error_tensor = np.square(y_hat - y_hat_f)
        else:
            error_tensor = np.square((y - y_f).squeeze(0).detach().numpy())

        if plot_dims is not None:
            assess_mse(error_tensor, file)
        mean_mse_channel = np.mean(error_tensor, axis=(1, 2))  # One MSE value per channel
        mse_tensor[:, i] = mean_mse_channel

        # Plot the dimensions which have the highest MSE across images
        if plot_dims is not None:
            versus_plot(original_img, test_img, main_title=f"Pixel Domain")
            for dim in plot_dims:
                versus_plot(y_hat[dim], y_hat_f[dim], main_title=f"Channel {dim}")

        """ Plot mean MSE over all channels + Histogram of MSE over channels """

        most_changed_channels[i, :] = np.argsort(mean_mse_channel)[-n_assess_dims:][::-1]

    return mse_tensor, most_changed_channels


def plot_max_mse_channels(max_mse_channels: np.ndarray, n_channels: int = 20,
                          top_threshold: float = 0.8, plot_title: str = "MSE Analysis of Latent Channels"):
    # Compute histogram
    sample_size = max_mse_channels.shape[0]
    # Make to list of channel ics, plot as bar plot
    unique_values, counts = np.unique(max_mse_channels.flatten(), return_counts=True)

    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(hspace=1, top=0.8)
    plt.subplot(3, 1, 1)
    plt.bar(unique_values, counts, align='center', width=0.5)  # Set width=0.5 for each bar
    plt.xlabel('Channel')
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel(f'Frequency in Top {n_channels}')
    plt.title(f"Times of occurences of a channel in the Top {n_channels} regarding max MSE\nOver {sample_size} samples")
    plt.xticks(unique_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot histogram over Frequency of occurence
    plt.subplot(3, 1, 2)
    plt.hist(counts, bins=[10*i for i in range(sample_size//10 + 1)])
    plt.title(f"Counts of occurence in Top {n_channels}")

    # Plot histogram over Frequency of occurence, bins are 5% Marker
    plt.subplot(3, 1, 3)
    plt.hist(counts, bins=20)
    plt.title(f"Counts of occurence in Top {n_channels} in 5% bins")


    plt.suptitle(plot_title, y=0.98)
    plt.show()
    top_channels = unique_values[np.where(counts >= top_threshold * sample_size)]  # Add a bit of tolerance
    print(f"The top channels regarding max MSE (In Top{n_channels} channels for >{top_threshold * 100}% of {sample_size} samples) are: \n{top_channels})")


if __name__ == "__main__":
    test_img_path = "../images/wallpaper.jpg"
    work_dir = "../images/own_manipulations_COCO/edited_images/"
    orig_path = "../images/own_manipulations_COCO/sheep/"
    fraud_img_files = os.listdir(work_dir)
    test_img_paths = [work_dir + s for s in fraud_img_files if s.endswith(".jpg")]
    original_img_paths = [orig_path + s for s in fraud_img_files if s.endswith(".jpg")]
    cheng_model = cai.zoo.cheng2020_anchor(6, pretrained=True)
    cheng_model.update()

    n_assess = 20

    max_mse_channels = assess_manipulated_images(original_img_paths, test_img_paths, cheng_model,
                                                 n_assess_dims=n_assess, plot_dims=[179, 185, 55])

    plot_max_mse_channels(max_mse_channels, n_assess)
