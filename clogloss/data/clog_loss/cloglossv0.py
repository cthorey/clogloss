import copy
import datetime
import os
import re
import uuid
from collections import Counter

import numpy as np
from box import Box

import cv2
import imageio
import pandas as pd
from orm import dionysos
from PIL import Image
from tqdm import tqdm
from vegai.data.clog_loss import common
from vegai.data.common import info_to_database
from vegai.utils import s3


def batch_to_image(batch):
    n = len(batch)
    w, h = batch[0].shape
    arr = np.vstack((batch, np.zeros((100 - n, w, h))))
    arr = [arr[j * 10:j * 10 + 10] for j in range(10)]
    final = []
    for a in arr:
        final.append(a.ravel().reshape((w * 10, w)).T)
    return Image.fromarray(np.vstack(final).astype('uint8'))


def process_one(filename, size=64):
    vid = imageio.get_reader(filename, 'ffmpeg')
    imgs = []
    for i in range(vid.count_frames()):
        frame = vid.get_data(i)
        mask = cv2.inRange(frame, (255, 69, 0), (255, 180, 0))
        mask = np.argwhere(mask)
        xmin, xmax, ymin, ymax = mask[:, 0].min(), mask[:, 0].max(
        ), mask[:, 1].min(), mask[:, 1].max()
        img = Image.fromarray(
            frame[xmin:xmax, ymin:ymax, :]).convert('L').resize((size, size))
        imgs.append(np.array(img))
    return batch_to_image(imgs)


class DatasetConstructor(object):
    def __init__(self,
                 data_name,
                 dest_folder,
                 split,
                 remove_local_copy=True,
                 upload_to_s3=True,
                 **kwargs):
        self.data_name = data_name
        self.dest_folder = dest_folder
        self.dataset = Box(data=[], info={})
        self.dataset.info.data_created = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M")
        self.dataset.info.update(kwargs)
        self.image_id = 0
        self.split = split
        self.remove_local_copy = remove_local_copy
        self.upload_to_s3 = upload_to_s3
        self.bucket = s3.S3Bucket(
            bucket_name='drivendata-competition-clog-loss')

    def save(self, payload, fname, data_type='image'):
        common.dump_payload(payload=payload,
                            fname=fname,
                            data_type=data_type,
                            remove_local_copy=self.remove_local_copy,
                            upload_to_s3=self.upload_to_s3)

    def process_record(self, record):
        key = record['url'].replace("s3://drivendata-competition-clog-loss/",
                                    "")
        self.bucket.download(key, 'record.mp4')
        image = process_one('record.mp4')
        drecord = Box()
        drecord['id'] = self.image_id
        drecord.update(record)
        name = 'image_{}.jpg'.format(uuid.uuid4().hex)
        drecord.image_path = os.path.join(self.dest_folder, 'data', name)
        self.save(image, drecord.image_path, 'image')
        self.dataset.data.append(drecord)
        self.image_id += 1

    def dump_annotations(self):
        """
        Dump the annotations.json to disk.
        """
        fname = 'annotations'
        if self.split is not None:
            fname = 'annotations_{}'.format(self.split)
        fname = os.path.join(self.dest_folder, '{}.json'.format(fname))
        self.save(self.dataset, fname, "annotations")


def process_split(samples, data_name, split, overfit, **kwargs):
    dataset = DatasetConstructor(data_name,
                                 split=split,
                                 overfit=overfit,
                                 **kwargs)
    datas = []
    for idx, sample in enumerate(samples.to_dict('record')):
        if idx % 10 == 0:
            print('Processing {}/{}'.format(idx, len(samples)))
        try:
            drecord = dataset.process_record(sample)
        except Exception as e:
            print(e)
            continue
    dataset.dump_annotations()


def get_dataset_name():
    with dionysos.session_scope() as sess:
        names = sess.query(dionysos.Dataset.dataset_name).all()
        names = [
            n.dataset_name for n in names
            if bool(re.match('cloglossv0.*(?<!overfit)$', n.dataset_name))
        ]
    i = 0
    while True:
        proposal = 'cloglossv0d{}'.format(i)
        if proposal not in names:
            break
        i += 1
    return proposal


def get_split(total=100000, test_size=0.25):
    concats = []
    meta = pd.read_csv('/workdir/data/raw/clog_loss/train_metadata.csv')
    labels = pd.read_csv('/workdir/data/raw/clog_loss/train_labels.csv')
    final = labels.set_index('filename').join(meta.set_index('filename'))
    concats.append(final[final.micro])
    others = final[~final.micro]
    concats.append(others[others.stalled == 1])
    others = others[others.stalled == 0]
    concats.append(others.sample(total - len(pd.concat(concats))))
    final = pd.concat(concats)
    w = final.groupby('stalled').size().to_dict()
    final['w'] = [1 - w[f] / float(sum(w.values())) for f in final.stalled]
    train = final.sample(int(len(final) * (1 - test_size)), weights='w')
    return train, final[~final.index.isin(train.index)]


def create_dataset(data_name='auto',
                   overwrite=False,
                   overfit=False,
                   maintainer='clement',
                   total=10000,
                   **kwargs):
    if data_name == 'auto':
        data_name = get_dataset_name()
    if overfit:
        data_name = '{}_overfit'.format(data_name)
    print('-' * 50)
    print('DATASET_NAME: {}'.format(data_name))
    data_cat = 'clog_loss'
    train, validation = get_split(total)
    if overfit:
        train = train[:50]
        validation = validation[:50]
    dest_folder = common.init_folder(data_name, overwrite=overwrite)
    entry = Box()
    for split, data in [('train', train), ('validation', validation)]:
        print('Processing {}'.format(split))
        process_split(data,
                      dest_folder=dest_folder,
                      data_name=data_name,
                      split=split,
                      overwrite=overwrite,
                      overfit=overfit)
    nsamples = Box({'train': len(train), 'validation': len(validation)})
    print('Updating the database')
    info_to_database(data_cat=data_cat,
                     maintainer=maintainer,
                     description="Trolley safe decision",
                     data_name=data_name,
                     dest_folder=dest_folder,
                     overwrite=overwrite,
                     nb_training_samples=nsamples.train,
                     nb_validation_samples=nsamples.validation)
