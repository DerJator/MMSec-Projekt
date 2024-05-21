import compressai as cai
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from compressai.ops import compute_padding
from compressai.models import CompressionModel
import os
import regex as re


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

    return mean_mse_channel


def assess_manipulated_images(original_imgs: list[str], manipulated_imgs: list[str],
                              model: CompressionModel, plot_dims: list[int], n_assess_dims: int = 20,
                              quantize: bool = False, device: torch.device = "cpu"):
    """
    For every pair of original and manipulated image, compress using given model and compare
    MSE between channels of compression. For every image, collect the 20 channel indices of max MSE and return
    this result as an array shape(n_imgs, n_assess_dims)
    """
    most_changed_channels = np.zeros((len(manipulated_imgs), n_assess_dims))

    for i, (original_path, manip_path) in enumerate(zip(original_imgs, manipulated_imgs)):
        test_img = plt.imread(manip_path)
        original_img = plt.imread(original_path)
        test = test_img / 255.
        original = original_img / 255.

        file = re.search(r"/([^/]+)$", original_path).group(1)

        y_f, y_hat_f = compress_latent(test, model, device=device)
        y, y_hat = compress_latent(original, model, device=device)

        # Calculate MSE between latent channels
        if quantize:
            error_tensor = np.square(y_hat - y_hat_f)
        else:
            error_tensor = np.square((y - y_f).squeeze(0).detach().numpy())

        mean_mse_channel = assess_mse(error_tensor, file)

        # Plot the dimensions which have the highest MSE across images
        versus_plot(original_img, test_img, main_title=f"Pixel Domain")
        for dim in plot_dims:
            versus_plot(y_hat[dim], y_hat_f[dim], main_title=f"Channel {dim}")

        """ Plot mean MSE over all channels + Histogram of MSE over channels """

        most_changed_channels[i, :] = np.argsort(mean_mse_channel)[-n_assess_dims:][::-1]

    return most_changed_channels


def plot_max_mse_channels(max_mse_channels: np.ndarray, n_channels: int = 20):
    # Compute histogram
    sample_size = max_mse_channels.shape[0]
    unique_values, counts = np.unique(max_mse_channels.flatten(), return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(unique_values, counts, align='center', width=0.5)  # Set width=0.5 for each bar
    plt.xlabel('Channel')
    plt.ylabel(f'Frequency in Top {n_channels}')
    plt.title(f"Occurence of a channel in the top {n_channels} regarding max MSE\nSample size: {sample_size}")
    plt.xticks(unique_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot histogram over Frequency of occurence
    plt.subplot(1, 2, 2)
    plt.hist(counts)
    plt.title(f"Counts of occurence in Top {n_channels}")

    plt.show()
    top_channels = unique_values[np.where(counts == sample_size)]  # Add a bit of tolerance
    print(f"The top channels regarding max MSE are: {top_channels})")


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
