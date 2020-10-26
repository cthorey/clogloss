from __future__ import print_function

import json
import os
import time
import uuid
from functools import wraps
from importlib import import_module

import numpy as np
import yaml
from box import Box

from vegai.utils import dictionary, s3


class ModelLoadingError(RuntimeError):
    pass


MODEL_TABLE = 'modelzoo'
MODEL_INDEX = 'modelzoo_index'
ROOT_DIR = os.environ['ROOT_DIR']
CONFIG_KEY = 'training_config'
CONFIG_FOLDER = os.path.join(ROOT_DIR, CONFIG_KEY)


def get_config_path(model_task, model_name, config_name):
    return os.path.join(CONFIG_FOLDER, 'models', model_task, model_name,
                        '{}.yaml'.format(config_name))


def get_cfg(model_task, model_name, config_name):
    path = get_config_path(model_task, model_name, config_name)
    if not os.path.isfile(path):
        raise IOError
    with open(path, 'r') as f:
        cfg = Box.from_yaml(f)
    return cfg


def load_model(model_task, model_name, expname):
    """
    general function to load a model
    """
    module_path = os.path.join('vegai', 'models', model_task, model_name,
                               'model').replace('/', '.')
    try:
        model = import_module(module_path).Model(model_task=model_task,
                                                 model_name=model_name,
                                                 expname=expname)
    except:
        raise ModelLoadingError()
    model.network.eval()
    model.network.freeze()
    return model


class BaseModel(object):
    """
    Base class for the model
    """
    def __init__(self,
                 model_name: str,
                 model_task: str,
                 model_description: str = "",
                 expname: str = None,
                 model_type: str = 'models'):
        """
        Args:
            model_type (str): Type of the model, 'models'
            model_name (str): Name of your model
            model_task (str): Task of your model
            stage (str): training or prediction
            expname (str): Name of the experiment
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model_task = model_task
        self.model_description = model_description
        self.model_folder = os.path.join(ROOT_DIR, self.model_type,
                                         self.model_task, self.model_name)
        self.bucket = s3.S3Bucket(bucket_name='omatai-project')
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        self.expname = expname

    def load(self, expname):
        """
        Load a specfic experiment
        """
        experiment_file = os.path.join(self.model_folder,
                                       '{}_experiment.json'.format(expname))
        if not os.path.isfile(experiment_file):
            s3.download_from_bucket(self.bucket, experiment_file)
        with open(experiment_file, 'r') as f:
            self.experiment = Box(json.load(f))
