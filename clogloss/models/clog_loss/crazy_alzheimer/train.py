import os
import time

import fire
import numpy as np

import pytorch_lightning as pl
import torch
from orm import dionysos
from pl_bolts.callbacks import PrintTableMetricsCallback
from pl_bolts.loggers import TrainsLogger
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import trainer
from vegai.models import experiment
from vegai.models.clog_loss.crazy_alzheimer import (augmentation, inspector,
                                                    model)


def evaluate(expname, gpus=1):
    m = model.Model(expname=expname)
    tr = trainer.Trainer()
    results = tr.test(m.network)


def get_callbacks(cfg, output_dir):
    cbacks = []
    checkpoint_path = os.path.join(output_dir, cfg.CHECKPOINT.NAME)
    checkpoint = pl_callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_last=False,
                                              monitor=cfg.CHECKPOINT.MONITOR,
                                              mode=cfg.CHECKPOINT.MONITOR_MODE)
    cs = [
        pl_callbacks.EarlyStopping(monitor=cfg.CHECKPOINT.MONITOR,
                                   mode=cfg.CHECKPOINT.MONITOR_MODE,
                                   **cfg.EARLY_STOPPING),
        pl_callbacks.LearningRateLogger(),
        inspector.AnalysisCallback()
    ]
    return checkpoint, cs


def auto_lr_find(network):
    tr = trainer.Trainer(gpus=1)
    result = tr.lr_find(network)
    return result.suggestion()


def train(config_name, max_epochs=1, maintainer='clement', gpus=1):
    cfg = model.get_cfg(config_name)
    task = experiment.LightningExperiment(model_task="clog_loss",
                                          model_name="crazy_alzheimer",
                                          config_name=config_name)
    m = model.Model()
    m.build_network(cfg.to_dict())
    if cfg.AUTO_LR_FIND:
        lr = auto_lr_find(m.network)
        cfg.SOLVER.DEFAULT_LR = lr
        m.build_network(cfg.to_dict())
    expname = task.next_trial_name()
    logger = task.start(expname)
    logger.experiment.connect_configuration(cfg)
    output_dir = os.path.join(m.model_folder, expname)
    checkpoint, cbacks = get_callbacks(cfg, output_dir)
    tr = trainer.Trainer(gpus=gpus,
                         default_root_dir=output_dir,
                         max_epochs=max_epochs,
                         logger=logger,
                         checkpoint_callback=checkpoint,
                         callbacks=cbacks,
                         **cfg.TRAINER)
    tr.fit(m.network)
    results = tr.test()
    task.upload_checkpoints(checkpoint,
                            expname,
                            label2id=m.network.train_set.label2id,
                            **results)
    task.end(expname=expname,
             dataset_name=m.network.train_set.data_name,
             score_name="mcc",
             score=results["val_mcc"],
             split="validation")


def explore(name):
    advisor = experiment.Bender(model_task='clog_loss',
                                model_name='crazy_alzheimer',
                                exploration_name=name)
    config_name = advisor.suggest()
    train(config_name, **advisor.exploration.common)


if __name__ == '__main__':
    fire.Fire()
