import os
import random
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from first_compressions import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle

"""
CASIA v2.0 data set contains 7465 authentic images, 5123 manipulated images 
Sample Names:
  Tp_D_CRN_S_N_cha00063_art00014_11818.jpg
  {Tampering}_{[D]ifferent Images => Spliced, [S]ame image => Spliced}_{Techniques (letters)}_{categ12345 (src)}_{categ6421 (target)}_ID
"""

categories = ["ani", "arc", "art", "cha", "ind", "nat", "pla", "sec", "txt"]
split = ["Au", "Sp"]  # Authentic and spliced
CASIA_PATH = "../images/CASIA2/CASIA2.0_revised"
CASIA_AUTH = Path(CASIA_PATH, "Au")
CASIA_TP = Path(CASIA_PATH, "Tp")
N_AUTHENTIC = 7465
N_TAMPERED = 5123
N_AU_STD = 6427


class Casia2Dataset(Dataset):
    """ Dataset for CASIA v2.0 Dataset, separated in tampered and associated authentic base image """
    def __init__(self, base_dir, channels, transform=None, seed=2811):
        self.base_dir = base_dir
        self.transform = transform
        self.auth_dir = Path(base_dir, "Au")
        self.tp_dir = Path(base_dir, "Tp")
        print(f"{self.auth_dir=}\n{self.tp_dir=}")
        self.channels = channels
        self.seed = seed

        # Find standard sized images, s.t. all images have same size 384 x ...
        # => Reduces dataset size to 1379 tampered images but allows uniform input for model
        std_info_auth = Path(self.auth_dir, "standard_sized")
        std_info_tp = Path(self.tp_dir, "standard_sized")
        if not os.path.exists(std_info_auth) or not os.path.exists(std_info_tp):
            self.annotate_standard_size_imgs()

        # Read the samples which have the standard size
        with open(std_info_auth, 'r') as file:
            self.auth_imgs = [img_path[:-1] for img_path in file.readlines()]
        with open(std_info_tp, 'r') as file:
            self.tp_imgs = [img_path[:-1] for img_path in file.readlines()]

        tp_au_pairs_file = Path(self.tp_dir, "tp-au-pairs.p")
        print(f"pickle file: {tp_au_pairs_file}")

        if os.path.exists(tp_au_pairs_file):
            print("read from existing pairing")
            with open(tp_au_pairs_file, 'rb') as file:
                self.tp_src_imgs = pickle.load(file)
        else:
            print("create new pairing")
            self.tp_src_imgs = self.find_img_pairs()
            with open(tp_au_pairs_file, "wb") as file:
                pickle.dump(self.tp_src_imgs, file)

        for i, item in enumerate(self.tp_src_imgs.items()):
            print(f"{item=}")
            if i == 3:
                break
        print(len(self.tp_src_imgs))
        self.mode = "neg"

        # self.auth_by_cat, self.tp_paths_by_cat = self.paths_per_cat()
        self.output_pairs = None
        self.labels = None
        self.organize_output_pairs(mode="neg")

        self.size = len(self.tp_imgs)
        print(f"{self.size=}")

    def train_test_split(self, test):
        train_pairs, test_pairs, train_labels, test_labels = (
            train_test_split(self.output_pairs, self.labels, test_size=test, random_state=self.seed))

        return train_pairs, test_pairs, train_labels, test_labels

    def paths_per_cat(self):
        """
        Parse image names, order them in a list by category, each category holds its images.
        return one dict for authentic and one for spliced images, keys are categories.

        auth_img_dir: directory of authentic images
        spl_img_dir: directory of spliced images
        """
        authentic, tampered = defaultdict(list), defaultdict(list)

        for categ in categories:
            tp_by_cat = [img_name for img_name in self.tp_imgs if categ in img_name]

            # For all images of this category, get authentic source images
            src_imgs = self._find_src_imgs(tp_by_cat)
            tampered[categ].extend(tp_by_cat)
            authentic[categ].extend(src_imgs.values())

        return authentic, tampered

    def _find_src_imgs(self):
        """
        For a list of tampered image names which encode the source name of the authentic image they're based on,
        extract the category and id of the source image and fetch the corresponding image name from the list of
        authentic images.
        """
        src_files = {}
        for tp_name in self.tp_imgs:
            match = re.search(r"([a-z]+)([0-9]+)(?=_[a-z])", tp_name)  # Get category of modified image
            categ = match.group(1)
            id = match.group(2)

            auth_src_id = categ + '_' + id  # Search authentic source image

            for auth_img in self.auth_imgs:
                if auth_src_id in auth_img:
                    src_files[tp_name] = auth_img
                    break

        return src_files

    def annotate_standard_size_imgs(self):
        """
        Go through images in dir and add txt file which specifies which images are of size 384 x 256 (to compare without resizing)
        """
        for dir in [self.auth_dir, self.tp_dir]:
            txt = open(Path(dir, "standard_sized"), "w")
            print("s_s path:", Path(dir, "standard_sized"))

            # Annotate and flip images
            for img_name in os.listdir(dir):
                img_path = Path(dir, img_name)
                if not img_name.endswith(".jpg"):
                    continue
                img = plt.imread(img_path)

                if 384 in img.shape and 256 in img.shape:
                    txt.write(f"{img_path}\n")  # Write in standard_info file
                    self.rotate_img(img, img_path)
                    if img is not None:
                        plt.imsave(img_path, img)  # Rotate and re-save for later loading
            txt.close()

    def find_img_pairs(self):
        tp_src = {}  # Keys: Tp-Path, Value: Corresponding Au-Path

        for tp_name in os.listdir(self.tp_dir):
            if not tp_name.endswith(".jpg"):
                continue

            match = re.search(r"([a-z]+)([0-9]+)(?=_[a-z])", tp_name)  # Get category and id of modified image
            categ = match.group(1)
            id = match.group(2)

            auth_src_id = categ + '_' + id  # Search authentic source image

            for auth_name in os.listdir(self.auth_dir):
                auth_path = Path(self.auth_dir, auth_name)
                if auth_src_id not in auth_name:
                    continue
                # if auth_path not in tp_src.values:
                #     continue

                img = plt.imread(auth_path)
                self.rotate_img(img, path=auth_path)

                if 384 in img.shape and 256 in img.shape:
                    tp_src[Path(self.tp_dir, tp_name).as_posix()] = auth_path.as_posix()
                break

        return tp_src

    def rotate_img(self, img: np.ndarray, path: Path):
        if 384 in img.shape and 256 in img.shape:
            if img.shape[0] == 256 and img.shape[1] == 384:
                img = img.transpose(1, 0, 2)

    def organize_output_pairs(self, mode="neg"):
        """
        Use the class variables to give a list of Au-Tp paris for each Tp image.
        mode: 'neg' or 'pos'. 'neg' creates the pair list and adds Au-Tp, 'pos' uses the existing list and adds Au-Au and Tp-Tp
        """
        if mode == "neg":
            self.output_pairs = []
            self.labels = []
            for tp_path, auth_path in self.tp_src_imgs.items():
                self.output_pairs.append((tp_path, auth_path))
                self.labels.append((1, 0))

        elif mode == "neg_pos":
            if self.output_pairs is None or self.labels is None:
                raise RuntimeError("Output lists need to be initialized by a call with mode='pos'")

            n_neg = len(self.tp_src_imgs)
            au_au_pairs = self.generate_pairs(list(self.tp_src_imgs.values()), n_neg)
            tp_tp_pairs = self.generate_pairs(list(self.tp_src_imgs.keys()), n_neg)

            for i in range(n_neg):
                self.output_pairs.append(au_au_pairs[i])  # Add Au-Au pair to output
                self.labels.append((0, 0))
                self.output_pairs.append(tp_tp_pairs[i])  # Add Tp-Tp pair to output
                self.labels.append((1, 1))

    def generate_pairs(self, l, n):
        pairs = []
        i = 0
        shuffled = l.copy()
        random.shuffle(l)

        while len(pairs) < n:
            if l[i] != shuffled[i]:
                pairs.append((l[i], shuffled[i]))
            else:
                random.shuffle(shuffled)
                i = 0
            i += 1

        return pairs

    def __getitem__(self, ix):
        img1, img2 = self.output_pairs[ix]
        labels = self.labels[ix]

        img1 = plt.imread(img1)
        img2 = plt.imread(img2)

        return img1[self.channels], img2[self.channels], labels

    def __len__(self):
        return self.size


def get_src_imgs(tp_imgs: list[str], auth_imgs: list[str]):
    """
    For a list of tampered image names which encode the source name of the authentic image they're based on,
    extract the category and id of the source image and fetch the corresponding image name from the list of
    authentic images.
    """
    src_files = []
    for tp_name in tp_imgs:
        match = re.search(r"([a-z]+)([0-9]+)(?=_[a-z])", tp_name)  # Get category of modified image
        categ = match.group(1)
        id = match.group(2)

        auth_src_id = categ + '_' + id  # Search authentic source image

        for auth_img in auth_imgs:
            if auth_src_id in auth_img:
                src_files.append(auth_img)

        print(tp_name, auth_src_id)
        print(src_files[-1])

    return src_files


def get_auth_pairs_casia2(auth_img_dir: str | Path, n_pairs, seed: int = 42):
    """
    Returns list of all authentic images that are standard sized in the specified directory and couples them randomly for comparison.
    """
    with open(Path(auth_img_dir, "standard_sized"), 'r') as file:
        std_imgs = [img_path[:-1] for img_path in file.readlines()]

    random.seed(seed)
    sampled_images = random.sample(std_imgs, 2 * n_pairs)
    pairs = [(sampled_images[i], sampled_images[i + 1])
             for i in range(0, 2 * n_pairs, 2)]

    return pairs


def pair_plot(authentic_path, spliced_path):
    authentic = plt.imread(Path(CASIA_AUTH, authentic_path))
    spliced = plt.imread(Path(CASIA_TP, spliced_path))
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

    casia2_data = Casia2Dataset(CASIA_PATH)
    casia2_loader = DataLoader(casia2_data, batch_size=10, shuffle=True)
    i = 0
    for imgs in casia2_loader:
        # imgs: ([10, 384, 256, 3])
        for k in range(10):
            versus_plot(imgs[0][k], imgs[1][k], "Casia2 Data")
            # print(imgs[0][k], imgs[1][k])
        i += 1
        if i >= 2:
            break
    exit()
    # annotate_standard_size_imgs(CASIA_AUTH)

    cheng_model = cai.zoo.cheng2020_anchor(6, pretrained=True)
    cheng_model.update()

    img_pairs = list(zip(auth_list, tp_list))
    mse_tensor, most_changed_channels = assess_manipulated_images(img_pairs[:100], cheng_model, n_assess_dims=192,
                                                 quantize=False)

    plot_max_mse_channels(most_changed_channels[:, :20], n_channels=20)
