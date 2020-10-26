import numpy as np
from box import Box

from imgaug import augmenters as iaa

AUGMENTATIONS = Box()
AUGMENTATIONS.default = dict(iaa=iaa.Sometimes(
    .6,
    iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Flipud(0.5),  # flip vertical
            iaa.Sometimes(0.3, iaa.Dropout(p=[0.1, 0.2])),
            iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 2)))
        ],
        random_order=True)))
