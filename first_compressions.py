import compressai as cai
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from compressai.ops import compute_padding


def compress_latent(img: torch.Tensor, model, device: torch.device):
    """
    model: CompressionModel
    """
    h, w = img.shape[-2:]
    pad, unpad = compute_padding(in_h=h, in_w=w, min_div=2 ** 6)

    padded_img = F.pad(input=img, pad=pad, mode="constant", value=0)

    padded_img = padded_img.to(device)
    model = model.to(device)

    y = model.g_a(padded_img)  # latent space direkt nach Encoder
    y_hat = model.gaussian_conditional.quantize(y, mode="dequantize") # latent space nach Quantisierung

    return y, y_hat


if __name__ == "__main__":
    test_img_path = "./wallpaper.jpg"
    test_img = plt.imread(test_img_path)
    test = test_img / 255.

    test_img_torch = torch.from_numpy(test.copy()).float()
    test_img_torch = test_img_torch.permute((2, 0, 1))
    test_img_torch = test_img_torch.unsqueeze(0)
    print(test_img_torch.size())
    # cheng_model = cai.models.Cheng2020Anchor()
    cheng_model = cai.zoo.cheng2020_anchor(6, pretrained=True)
    cheng_model.update()

    y, y_hat = compress_latent(test_img_torch, cheng_model, device="cpu")
    print(type(y_hat))
    print(y.size())

    latent_channels = []

    y_hat = y_hat.squeeze(0).detach().cpu().numpy().copy()

    for i in range(y_hat.shape[0]):
        plt.imshow(y_hat[i], cmap="gray")
        plt.show()
