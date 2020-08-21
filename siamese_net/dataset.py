import random
from PIL import Image, ImageOps
import numpy as np

import torch
from torch.utils.data import Dataset

class SiameseNetworkDataset(Dataset):
    """
    Author: Alan Preciado
    """
    def __init__(self, image_folder_dataset, transform=None, should_invert=True):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        anchor_tuple = random.choice(self.image_folder_dataset.imgs)
        positive_tuple = self._get_img(*anchor_tuple, True)
        negative_tuple = self._get_img(*anchor_tuple, False)

        anchor_img = Image.open(anchor_tuple[0])
        positive_img = Image.open(positive_tuple[0])
        negative_img = Image.open(negative_tuple[0])

        if self.should_invert:
            anchor_img = ImageOps.invert(anchor_img)
            positive_img = ImageOps.invert(positive_img)
            negative_img = ImageOps.invert(negative_img)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def _get_img(self, img_path, class_id, should_get_same_class):
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        img_tuple = None
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img_tuple = random.choice(self.image_folder_dataset.imgs)
                if class_id == img_tuple[1] and img_path != img_tuple[0]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img_tuple = random.choice(self.image_folder_dataset.imgs)
                if class_id != img_tuple[1]:
                    break
        return img_tuple

    def __len__(self):
        return len(self.image_folder_dataset.imgs)
