import copy
import os

import numpy as np

from PIL import Image
from torchvision import transforms as T
from vegai.data.clog_loss import data

ROOT_DIR = os.environ['ROOT_DIR']


class IAAAugmenter(object):
    def __init__(self, augmentation):
        self.augmentation = augmentation

    def __call__(self, image, target=None):
        det = self.augmentation.to_deterministic()
        image = np.array(image)
        image = Image.fromarray(det.augment_image(image))
        if target is not None:
            image = image, target
        return image


def build_transforms(cfg, stage, augmentation=None):
    crop_size = cfg.INPUT.CENTER_CROP_SIZE
    to_compose = []
    if augmentation and stage == 'train':
        if 'iaa' in augmentation:
            aug = IAAAugmenter(augmentation['iaa'])
            to_compose.append(aug)
        if 'transforms' in augmentation:
            to_compose += augmentation['transforms']
    to_compose += [
        T.CenterCrop(size=crop_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ]
    return T.Compose(to_compose)


class TrainingDataset(data.MosaicDataset):
    def __init__(self, data_name, split, transforms):
        super(TrainingDataset, self).__init__(data_name, split)
        self.transforms = transforms

    def __getitem__(self, idx):
        record = self.get_img_info(idx)
        target = record['stalled']
        x = self.get_images(idx)
        x = self.transforms(x.convert('RGB'))
        return x, target
