import numpy as np
import matplotlib.pyplot as plt
import Casia2
import os
from pathlib import Path
import first_compressions
import compressai as cai
import cv2
import torch
import torch.nn as nn
from torchvision import models


class VGG19FeatExtract(nn.Module):
    def __init__(self):
        vgg19 = models.vgg19(pretrained=True)
        print(vgg19)
        super(VGG19FeatExtract, self).__init__()
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            *list(vgg19.classifier.children())[:-3]  # Retain layers up to the second last fully connected layer
        )
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg19_feature_extractor():
    vgg19 = models.vgg19(pretrained=True)
    print(vgg19)

    # Remove the last fully connected layer (not interesting for feat extraction)
    vgg19.classifier = nn.Sequential(*list(vgg19.classifier.children())[:-1])
    # print(vgg19)


def extract_cooccurrences(in_tensor: np.ndarray, channels: [int], T=3, Q=5) -> list:
    # in_tensor: [n_channels, m, n]
    n_c, m, n = in_tensor.shape
    img_median = np.median(in_tensor)

    img_bordered = np.zeros((n_c, m + 2, n + 2))
    img_bordered[:, 1:m+1, 1:n+1] = in_tensor
    img_bordered[:, :, 0] = np.repeat(img_median, (m+2) * n_c).reshape((n_c, m+2))
    img_bordered[:, :, n+1] = np.repeat(img_median, (m+2) * n_c).reshape((n_c, m+2))
    img_bordered[:, 0, :] = np.repeat(img_median, (n+2) * n_c).reshape((n_c, n+2))
    img_bordered[:, m+1, :] = np.repeat(img_median, (n+2) * n_c).reshape((n_c, n+2))

    hists = []

    for c in channels:
        cooc_feat = {}
        plt.hist(img_bordered[c].flatten())
        plt.show()
        for i in range(1, m+1):
            for j in range(1, n+1):
                cooc = [img_bordered[c, i - 1, j],
                        img_bordered[c, i, j + 1],
                        img_bordered[c, i + 1, j],
                        img_bordered[c, i, j - 1]]

                print(cooc)

                cooc = tuple(np.clip(round(val/Q), -T, T) for val in cooc)
                print(cooc)

                # Merge symmetric co-occurrences
                if cooc_feat.get(cooc) is not None:
                    cooc_feat[cooc] += 1
                elif cooc_feat.get(cooc[::-1]) is not None:
                    cooc_feat[cooc[::-1]] += 1
                else:
                    cooc_feat[cooc] = 1
                # print(cooc_feat)
        hists.append(cooc_feat)

    return hists


if __name__ == '__main__':
    test_img = np.random.rand(4, 64, 64) * 255
    cheng_model = cai.zoo.cheng2020_anchor(6, pretrained=True)

    for img_path in os.listdir(Casia2.CASIA_TP):
        test_img = plt.imread(Path(Casia2.CASIA_TP, img_path))
        img_low = cv2.GaussianBlur(test_img, (5, 5), 4)
        hp_channel = test_img - img_low

        y, y_q = first_compressions.compress_latent(test_img / np.max(test_img), cheng_model, device="cpu")
        detached_latent = y.squeeze(0).detach().numpy()

        hp_channel = np.zeros_like(detached_latent)
        for i in range(detached_latent.shape[0]):
            img_low = cv2.GaussianBlur(detached_latent[i], (5, 5), 1)
            hp_channel[i] = detached_latent[i] - img_low

        res = extract_cooccurrences(hp_channel, [i for i in range(50)], T=6, Q=2)

        for d in res:
            print(d.keys())
        break

