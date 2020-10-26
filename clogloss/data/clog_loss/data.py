import hashlib
import json
import os
import pickle
import random
from collections import Counter

import numpy as np
from box import Box

from matplotlib import pylab as plt
from PIL import Image
from torch.utils.data import Dataset

ROOT_DIR = os.environ['ROOT_DIR']
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'interim')


class MosaicDataset(Dataset):
    def __init__(self, data_name, split):
        self.data_name = data_name
        self.split = split
        fname = os.path.join(DATA_PATH, data_name,
                             'annotations_{}.json'.format(split))
        with open(fname) as f:
            self.dataset = Box(json.load(f))
        self._index()
        self.classes = ['stalled', 'clear']
        self.label2id = Box({'stalled': 0, 'clear': 1})

    def __len__(self):
        return len(self.ids)

    @property
    def dataset_info(self):
        """
        Additional info for uniquely indentifying a dataset
        """
        info = Box(data_name=self.data_name,
                   image_type=self.image_type,
                   classes=self.classes)
        return info

    @property
    def dataset_id(self):
        """
        Provide a unique ID for this dataset
        """
        return hashlib.sha256(self.dataset_info.to_json()).hexdigest()

    @property
    def id2label(self):
        return {v: k for k, v in self.label2id.items()}

    @property
    def support(self):
        d = dict(
            Counter([self.get_img_info(idx).stalled
                     for idx in self.image_ids]))
        return Box(d)

    @property
    def image_ids(self):
        return self.ids

    def _index(self):
        self.ids = [d['id'] for d in self.dataset.data]

    def get_img_info(self, idx):
        return self.dataset.data[idx]

    def display(self, idx):
        print('*' * 50)
        print('Decision: {}'.format(self[idx][1]))
        print('*' * 50)
        img = self[idx][0]
        return img

    def display_random(self):
        idx = random.choice(self.ids)
        return self.display(idx)

    def get_images(self, idx):
        record = self.get_img_info(idx)
        return Image.open(record['image_path'])

    def __getitem__(self, idx):
        record = self.get_img_info(idx)
        img = self.get_images(idx)
        target = record['stalled']
        return img, target
