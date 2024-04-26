import compressai as cai
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from compressai.ops import compute_padding
import os


def compress_latent(img: np.ndarray, model, device: torch.device):
    """
    model: CompressionModel
    """
    img = img_to_torch(img)

    h, w = img.shape[-2:]
    pad, unpad = compute_padding(in_h=h, in_w=w, min_div=2 ** 6)

    padded_img = F.pad(input=img, pad=pad, mode="constant", value=0)

    padded_img = padded_img.to(device)
    model = model.to(device)

    y = model.g_a(padded_img)  # latent space direkt nach Encoder
    y_hat = model.gaussian_conditional.quantize(y, mode="dequantize") # latent space nach Quantisierung

    return y, y_hat


def img_to_torch(img: np.ndarray) -> torch.Tensor:
    img = torch.from_numpy(img.copy()).float()
    img = img.permute((2, 0, 1))
    img = img.unsqueeze(0)

    return img


def versus_plot(img1: np.ndarray, img2: np.ndarray, main_title: str,
                title1: str = "Original", title2: str = "Manipulated"):
    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    # Plotting the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.imshow(img1, cmap='gray')  # Adjust the colormap if necessary
    plt.title(title1)

    # Plotting the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.imshow(img2, cmap='gray')  # Adjust the colormap if necessary
    plt.title(title2)

    plt.show()


if __name__ == "__main__":
    test_img_path = "./images/wallpaper.jpg"
    work_dir = "./images/own_manipulations_COCO/edited_images/"
    original_path = "./images/own_manipulations_COCO/sheep/"
    fraud_img_files = os.listdir(work_dir)
    test_img_paths = [work_dir + s for s in fraud_img_files if s.endswith(".jpg")]
    original_img_paths = [original_path + s for s in fraud_img_files if s.endswith(".jpg")]
    cheng_model = cai.zoo.cheng2020_anchor(6, pretrained=True)
    cheng_model.update()

    most_changed_channels = np.zeros((len(test_img_paths), 20))

    # Loop over pairs of original-manipulated images
    for i, (original_path, test_path) in enumerate(zip(original_img_paths, test_img_paths)):
        test_img = plt.imread(test_path)
        original_img = plt.imread(original_path)
        test = test_img / 255.
        original = original_img / 255.

        y_f, y_hat_f = compress_latent(test, cheng_model, device="cpu")
        y, y_hat = compress_latent(original, cheng_model, device="cpu")

        y_hat_f = y_hat_f.squeeze(0).detach().cpu().numpy().copy()
        y_hat = y_hat.squeeze(0).detach().cpu().numpy().copy()
        error_tensor = np.zeros_like(y_hat)

        # Plot the dimensions which have the highest MSE across images
        versus_plot(y_hat[179], y_hat_f[179], main_title="Channel 179")
        versus_plot(y_hat[185], y_hat_f[185], main_title="Channel 185")
        versus_plot(y_hat[125], y_hat_f[125], main_title="Channel 125")

        for channel in range(y_hat.shape[0]):
            # Calculate MSE between latent channels
            error_tensor[channel] = np.square(y_hat[channel] - y_hat_f[channel])

        """ Plot mean MSE over all channels + Histogram of MSE over channels """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"MSE original vs. manipulated over latent\nimg{i}")
        plt.imshow(np.mean(error_tensor, axis=0), cmap="plasma")

        mean_mse_channel = np.mean(error_tensor, axis=(1, 2))
        plt.subplot(1, 2, 2)
        plt.title(f"Mean MSE per channel\nimg{i}, argmax: {np.argmax(mean_mse_channel)}")
        plt.bar(np.arange(mean_mse_channel.shape[0]), mean_mse_channel)
        plt.show()

        most_changed_channels[i, :] = np.argsort(mean_mse_channel)[-20:][::-1]

    # Compute histogram
    sample_size = most_changed_channels.shape[0]
    unique_values, counts = np.unique(most_changed_channels.flatten(), return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(unique_values, counts, align='center', width=0.5)  # Set width=0.5 for each bar
    plt.xlabel('Channel')
    plt.ylabel('Frequency in Top 20')
    plt.title(f"Occurence of a channel in the top 20 regarding max MSE\nSample size: {sample_size}")
    plt.xticks(unique_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot histogram over Frequency of occurence
    plt.subplot(1, 2, 2)
    plt.hist(counts)
    plt.title("Counts of occurence in Top 20")

    plt.show()

    top_channels = unique_values[np.where(counts == sample_size)]  # Add a bit of tolerance
    print(top_channels)
