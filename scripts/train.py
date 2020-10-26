#!/usr/bin/env python2
import os
import sys
import uuid

import docker
import fire

VOLUMES = [
    '~/.aws:/root/.aws:rw', '~/.pgpass:/root/.pgpass:rw',
    '~/workdir/training_config:/workdir/training_config:rw',
    '~/.trains.conf:/root/trains.conf:rw',
    '/mnt/hdd/omatai/data/interim:/workdir/data/interim:rw',
    '/mnt/hdd/omatai/models:/workdir/models:rw'
]
ROOT_DIR = os.path.join(os.environ['HOME'], 'workdir', 'omatai')
docker_client = docker.from_env()


def train(config_name,
          model_task,
          model_name,
          image_tag='latest',
          devices=0,
          max_epochs=30,
          maintainer='clement',
          detach=True):
    sys.path.append('./scripts')
    import docker_utils

    cmd = docker_utils.parse_cmd(
        script="/packages/vegai/models/{}/{}/train.py".format(
            model_task, model_name),
        fn='train',
        config_name=config_name,
        max_epochs=max_epochs,
        maintainer=maintainer,
        gpus=1)
    print(cmd)
    envs = docker_utils.create_envs(devices)
    volumes = docker_utils.create_volumes(VOLUMES)
    name = '{}_{}'.format(config_name, uuid.uuid4().hex)
    remove = False if not detach else True
    docker_client.containers.run(image='xihelm/vegai:{}'.format(image_tag),
                                 name=name,
                                 runtime='nvidia',
                                 command=cmd,
                                 detach=detach,
                                 stdout=True,
                                 environment=envs,
                                 remove=remove,
                                 volumes=volumes,
                                 shm_size="32G")


def explore(model_task,
            model_name,
            name,
            image_tag='latest',
            devices=0,
            detach=True):
    sys.path.append('./scripts')
    import docker_utils

    cmd = docker_utils.parse_cmd(
        script="/packages/vegai/models/{}/{}/train.py".format(
            model_task, model_name),
        fn='explore',
        name=name)
    envs = docker_utils.create_envs(devices)
    volumes = docker_utils.create_volumes(VOLUMES)
    name = '{}_{}'.format(name, uuid.uuid4().hex)
    remove = False if not detach else True
    policy = docker_utils.create_policy("unless-stopped")
    docker_client.containers.run(image='xihelm/vegai:{}'.format(image_tag),
                                 name=name,
                                 restart_policy=policy,
                                 runtime='nvidia',
                                 command=cmd,
                                 detach=detach,
                                 stdout=True,
                                 environment=envs,
                                 volumes=volumes,
                                 shm_size="32G")


if __name__ == '__main__':
    fire.Fire()
