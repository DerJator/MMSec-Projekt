import os
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from first_compressions import *

"""
CASIA v1.0 data set contains 800 authentic images, 462 spliced images 
"""

categories = ["ani", "arc", "art", "cha", "nat", "pla", "sec", "txt"]
split = ["Au", "Sp"] # Authentic and spliced
casia_path = "../images/CASIA1/"
auth_ext = "Au"
splice_ext = "Modified_Sp/Sp"
cm_ext = "Modified_Sp/CM"


def get_source_img(mod_name: str, categ: str, auth_imgs: list):
    src_categ = re.search(categ + r"[0-9]*", mod_name).group(0)
    src_categ_mod = src_categ[:3] + '_' + src_categ[3:]  # Authentic images add underscore

    for auth_img in auth_imgs:
        if src_categ_mod in auth_img:
            return auth_img


def parse_names(auth_img_dir: Path, spl_img_dir: Path):
    """
    Parse image names, order them in a list by category, each category holds its images.
    return one list for authentic and one for spliced images.

    auth_img_dir: directory of authentic images
    spl_img_dir: directory of spliced images
    """
    spl_imgs = os.listdir(spl_img_dir)
    auth_imgs = os.listdir(auth_img_dir)
    authentic, spliced = defaultdict(list), defaultdict(list)
    for spl_title in spl_imgs:

        for categ in categories:
            if not categ in spl_title:
                continue

            # Append to authentic , find counterpart
            auth_title = get_source_img(spl_title, categ, auth_imgs)
            authentic[categ].append(Path(auth_img_dir, auth_title).as_posix())
            spliced[categ].append(Path(spl_img_dir, spl_title).as_posix())

    return authentic, spliced


def pair_plot(authentic_path, spliced_path):
    authentic = plt.imread(Path(casia_path, auth_ext, authentic_path))
    spliced = plt.imread(Path(casia_path, splice_ext, spliced_path))
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Display the first image
    axes[0].imshow(authentic)
    axes[0].axis('off')  # Hide the axes
    axes[0].set_title('Authentic')

    # Display the second image
    axes[1].imshow(spliced)
    axes[1].axis('off')  # Hide the axes
    axes[1].set_title('Spliced')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Spliced data, 462 Samples
    auth_data, spl_data = parse_names(Path(casia_path, auth_ext), Path(casia_path, splice_ext))
    # Copy-Move, 459 Samples
    cm_auth_data, cm_data = parse_names(Path(casia_path, auth_ext), Path(casia_path, cm_ext))

    cheng_model = cai.zoo.cheng2020_anchor(6, pretrained=True)
    cheng_model.update()

    for (key_auth, auth_imgs), (key_spl, spl_imgs) in zip(auth_data.items(), spl_data.items()):
        max_mse_channels = assess_manipulated_images(auth_imgs, spl_imgs, cheng_model, n_assess_dims=20, plot_dims=[179, 185, 55],
                                                     quantize=False)
        plot_max_mse_channels(max_mse_channels, n_channels=20)
