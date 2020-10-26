from __future__ import print_function
import pandas as pd
from functools import wraps
import hashlib
import json
import os
import time
import uuid
from functools import wraps
import datetime
import numpy as np
import yaml
from box import Box
import torch
from orm.dionysos import VegaiModelzoo, session_scope
from pl_bolts.loggers import TrainsLogger
from vegai.utils import dictionary, s3
from vegai.models import common
ROOT_DIR = os.environ['ROOT_DIR']


def spin_on_error(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        print('*' * 50)
        print('TrainingLand')
        print('*' * 50)
        try:
            function(*args, **kwargs)
        except Exception as e:
            print(e)
            print('sleeping now')
            time.sleep(10000000)


def retrieve_summary(model_task='.*',
                     model_name='.*',
                     dataset_name='.*',
                     expname='.*',
                     score_name='.*',
                     split='.*',
                     model_id=None):
    """
    Retrieve previously computed summary
    """
    with session_scope() as sess:
        query = sess.query(VegaiModelzoo).\
                filter(VegaiModelzoo.model_name.op('~')(model_name)).\
                filter(VegaiModelzoo.dataset_name.op('~')(dataset_name)).\
                filter(VegaiModelzoo.model_task.op('~')(model_task)).\
                filter(VegaiModelzoo.score_name.op('~')(score_name)).\
                filter(VegaiModelzoo.split.op('~')(split)).\
                filter(VegaiModelzoo.expname.op('~')(expname))
        if model_id is not None:
            query = query.filter(VegaiModelzoo.model_id == model_id)
        records = query.all()
        entries = []
        for record in records:
            record = Box(record.__dict__)
            record.pop('_sa_instance_state')
            entries.append(record)
    return pd.DataFrame(entries)


class Experiment(object):
    def __init__(self, model_task: str, model_name: str, config_name: str):
        self.model_task = model_task
        self.model_name = model_name
        self.config_name = config_name
        self.bucket = s3.S3Bucket(bucket_name='omatai-project')

    def get_model_id(self, expname):
        entry = dict(model_task=self.model_task,
                     model_name=self.model_name,
                     expname=expname)
        return hashlib.sha1(json.dumps(entry).encode()).hexdigest()

    def next_trial_name(self):
        with session_scope() as sess:
            results = sess.query(VegaiModelzoo).filter(
                VegaiModelzoo.model_task == self.model_task).filter(
                    VegaiModelzoo.model_name == self.model_name).filter(
                        VegaiModelzoo.expname.op('~')("{}.*".format(
                            self.config_name))).all()
            expnames = [r.expname for r in results]
            if not expnames:
                idx = 0
            else:
                idx = max(
                    [int(expname.split('t')[-1]) for expname in expnames])
        expname = "{}t{}".format(self.config_name, idx + 1)
        return expname

    def log_task(self, expname: str):
        with session_scope() as sess:
            entry = dict(model_task=self.model_task,
                         model_name=self.model_name,
                         expname=expname)
            model_id = self.get_model_id(expname)
            entry = VegaiModelzoo(
                model_id=model_id,
                status="started",
                created_on=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                **entry)
            sess.add(entry)

    def update_task(self, expname, **kwargs):
        with session_scope() as sess:
            entry = dict(model_task=self.model_task,
                         model_name=self.model_name,
                         expname=expname)
            model_id = hashlib.sha1(json.dumps(entry).encode()).hexdigest()
            entry = sess.query(VegaiModelzoo).filter(
                VegaiModelzoo.model_id == model_id).all()[0]
            for key, value in kwargs.items():
                setattr(entry, key, value)

    def start(self, expname: str):
        self.log_task(expname)

    def end(self, expname: str, status: str):
        self.update_task(expname, status=status)


class LightningExperiment(Experiment):
    def start(self, expname: str):
        self.log_task(expname)
        task = TrainsLogger(project_name=self.model_task,
                            task_name="{}/{}".format(self.model_name, expname),
                            reuse_last_task_id=False)
        return task

    def upload_checkpoints(self, checkpoint, expname, **kwargs):
        checkpoint_path = checkpoint.best_model_path
        if checkpoint_path == "":
            raise ValueError('Could not retrieve checkpoints')
        # add some stuff in there
        data = torch.load(checkpoint_path)
        data['extra'] = kwargs
        torch.save(data, checkpoint_path)
        # upload to s3
        key = 'models/{}/{}/{}_weights.pth'.format(self.model_task,
                                                   self.model_name, expname)
        print('Uploading {} to s3'.format(key))
        self.bucket.upload_from_file(checkpoint_path, key, overwrite=True)

    def end(self,
            expname: str,
            dataset_name: str,
            score_name: str,
            score,
            split: str = "validation",
            status: str = "success",
            maintainer: str = "clement"):
        self.update_task(expname,
                         dataset_name=dataset_name,
                         score_name=score_name,
                         score=score,
                         status=status,
                         split=split,
                         maintainer=maintainer)


class Bender(object):
    def __init__(self, model_task, model_name, exploration_name):
        self.config_folder = os.path.join(common.CONFIG_FOLDER, 'models',
                                          model_task, model_name)
        self.exploration_name = exploration_name
        path = os.path.join(self.config_folder, 'explorations',
                            '{}.yaml'.format(exploration_name))
        self.exploration = Box.from_yaml(open(path, 'r'))

    def suggest_params(self):
        params = dict()
        parameter_space = self.exploration.parameter_space.to_dict()
        for key, param in parameter_space.items():
            if param['type'] == 'params':
                probs = param.get('probs', None)
                values = param['values']
                p = dict(zip(range(len(values)), values))
                value = p[np.random.choice(list(p.keys()), p=probs).item()]
                params[param['name']] = value
            elif param['type'] == 'constant':
                params[param['name']] = param['value']
            else:
                raise RuntimeError('Wrong parameter type')
        return params

    def suggest(self):
        base_cfg = self.exploration.base_cfg.to_dict()
        base = dictionary.flatten(base_cfg, sep='.')
        params = self.suggest_params()
        base.update(params)
        new_cfg = dictionary.unflatten(base, sep='.')
        cfg_name = '{}_{}'.format(self.exploration_name, uuid.uuid4().hex)
        with open(os.path.join(self.config_folder, '{}.yaml'.format(cfg_name)),
                  'w+') as f:
            yaml.dump(new_cfg, f)
        return cfg_name
