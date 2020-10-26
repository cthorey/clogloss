"""
Example template for defining a system.
"""

import numpy as np
from box import Box

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics import ConfusionMatrix
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from vegai.models import losses
from vegai.models.clog_loss.crazy_alzheimer import augmentation as augs
from vegai.models.clog_loss.crazy_alzheimer import data, layers

OPTIMIZER = dict(Adam=optim.Adam, RMSprop=optim.RMSprop)
SCHEDULER = dict(OneCycleLR=optim.lr_scheduler.OneCycleLR)
LOSSES = dict(CrossEntropyLoss=losses.CrossEntropyLoss)
BACKBONES = dict(resnet18=models.resnet18, resnet50=models.resnet50)


def metrics_from_cm(tp, tn, fp, fn):
    mcc = (tp * tn - fp * fn) / torch.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    precision = tp / (tp + fn)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        'data': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        },
        'mcc': mcc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'acc': acc
    }


def metrics(logits, targets):
    preds = torch.argmax(logits, dim=1)
    cm = ConfusionMatrix()(preds, targets)
    if len(cm.size()) == 0:
        idx = preds[0].item()
        n = cm.item()
        cm = torch.zeros((2, 2))
        cm[idx, idx] = n
    # cm_{i,j} is the number of observations in group i that were predicted in group j
    tp, tn, fn, fp = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
    metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    return metrics


def build_backbone(cfg, is_train=True):
    cc = cfg.MODEL
    m = BACKBONES[cc.BACKBONE.NAME]
    download_pretrained = cc.BACKBONE.USE_PRETRAINED
    if not is_train:
        download_pretrained = False
    branch = m(download_pretrained)
    if cc.BACKBONE.FREEZE and is_train:
        branch.requires_grad_(False)
    # cut out the head of resnet
    ll = list(branch.children())[:-2]
    for idx in cc.BACKBONE.UNFREEZE_LAYERS:
        for p in ll[idx].parameters():
            p.requires_grad_(True)
    # add a layer to normalize the output
    ll += [layers.AdaptiveConcatPool2d()]
    return nn.Sequential(*ll)


class MosaicNetwork(pl.LightningModule):
    def __init__(self, cfg, is_train=True):
        super().__init__()
        if isinstance(cfg, Box):
            raise ValueError('Pass a dict instead')
        self.save_hyperparameters('cfg', 'is_train')
        self.cfg = Box(cfg)
        self.batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        self.learning_rate = self.cfg.SOLVER.DEFAULT_LR
        self.add_module('backbone', build_backbone(self.cfg,
                                                   is_train=is_train))
        cc = self.cfg.MODEL.HEAD
        lin_ftrs = cc.NFEATURES if isinstance(cc.NFEATURES,
                                              list) else [cc.NFEATURES]
        nf = layers.num_features_model(self.backbone)
        head = layers.create_head(nf,
                                  self.cfg.DATASETS.NUM_CLASSES,
                                  lin_ftrs,
                                  use_pool=False)
        self.add_module('head', head)
        self.criterion = LOSSES[self.cfg.SOLVER.LOSS.NAME](
            **self.cfg.SOLVER.LOSS.get('PARAMS', {}))
        self.augmentation = None
        if cfg.get('AUGMENTATION', ''):
            self.augmentation = augs.AUGMENTATIONS[cfg.get('AUGMENTATION', '')]

    def setup(self, stage):
        transforms = data.build_transforms(self.cfg,
                                           stage='train',
                                           augmentation=self.augmentation)
        self.train_set = data.TrainingDataset(transforms=transforms,
                                              **self.cfg.DATASETS.TRAIN)
        transforms = data.build_transforms(self.cfg, stage='validation')
        self.validation_set = data.TrainingDataset(transforms=transforms,
                                                   **self.cfg.DATASETS.TEST)
        transforms = data.build_transforms(self.cfg, stage='test')
        self.test_set = data.TrainingDataset(transforms=transforms,
                                             **self.cfg.DATASETS.TEST)

    @pl.data_loader
    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.cfg.DATALOADER.CLASS_BALANCING:
            total = float(sum(self.train_set.support.values()))
            id2weight = {
                k: 1 - float(v) / float(total)
                for k, v in self.train_set.support.items()
            }
            targets = [
                id2weight[self.train_set.get_img_info(idx)['stalled']]
                for idx in self.train_set.image_ids
            ]
            w = torch.Tensor(targets).double()
            sampler = torch.utils.data.sampler.WeightedRandomSampler(w, len(w))
            shuffle = False
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          sampler=None,
                          num_workers=self.cfg.DATALOADER.NUM_WORKERS)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.validation_set,
                          batch_size=self.batch_size,
                          num_workers=self.cfg.DATALOADER.NUM_WORKERS)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.cfg.DATALOADER.NUM_WORKERS)

    def forward(self, x):
        x = torch.flatten(self.backbone(x), start_dim=1)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        # forward pass
        images, targets = batch
        preds = self(images)
        loss = self.criterion(preds, targets)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        """
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        logs = metrics(logits, targets)
        logs['val_loss'] = loss
        return logs

    def validation_epoch_end(self, outputs):
        cm = dict()
        logs = dict()
        for metric_name in outputs[0].keys():
            if metric_name == 'val_loss':
                logs[metric_name] = torch.stack(
                    [x[metric_name] for x in outputs]).mean()
            else:
                cm[metric_name] = torch.stack(
                    [x[metric_name] for x in outputs]).sum()
        perf = metrics_from_cm(cm['tp'], cm['tn'], cm['fp'], cm['fn'])
        logs.update(
            {'val_{}'.format(k): v
             for k, v in perf.items() if k != 'data'})
        result = {
            'data': perf['data'],
            'progress_bar': logs,
            'log': logs,
            'val_loss': logs["val_loss"]
        }
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        params = self.cfg.SOLVER.OPTIMIZER.get('PARAMS', {})
        params['lr'] = self.learning_rate
        optimizer = OPTIMIZER[self.cfg.SOLVER.OPTIMIZER.NAME](
            self.parameters(), **dict(params))
        params = self.cfg.SOLVER.SCHEDULER.get('PARAMS', {})
        if self.cfg.SOLVER.SCHEDULER.NAME == 'OneCycleLR':
            params['max_lr'] = self.learning_rate
        scheduler_lr = SCHEDULER[self.cfg.SOLVER.SCHEDULER.NAME](
            optimizer, **dict(params))
        scheduler = self.cfg.SOLVER.SCHEDULER_META
        scheduler['scheduler'] = scheduler_lr

        return [optimizer], [scheduler]
