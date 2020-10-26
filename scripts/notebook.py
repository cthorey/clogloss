#!/usr/bin/env python
import os
import sys
import uuid

import fire

IMAGE_TAG = 'latest'


def launch(port=8888):
    """
    Launch a container which call the method upload within itself
    """
    sys.path.append('./scripts')
    import docker_utils
    import docker

    def create_volumes(entrypoint=None):
        vs = [
            '~/.aws:/root/.aws:rw', '~/.pgpass:/root/.pgpass:rw',
            '~/.trains.conf:/root/trains.conf:rw',
            '/tmp/.X11-unix:/tmp/.X11-unix:rw',
            '~/workdir/vegai/vegai:/packages/vegai:rw',
            '~/workdir/vegai/scripts:/workdir/scripts:rw',
            '~/workdir/vegai/notebooks:/workdir/notebooks:rw',
            '~/workdir/orm:/packages/orm:rw',
            '~/workdir/training_config:/workdir/training_config:rw',
            '/mnt/hdd/omatai/data/interim:/workdir/data/interim:rw',
            '/mnt/hdd/omatai/data/raw:/workdir/data/raw:rw',
            '/mnt/hdd/omatai/models:/workdir/models:rw'
        ]
        if entrypoint:
            vs.append('{}:/workdir/entrypoint.sh:rw'.format(entrypoint))
        return docker_utils.create_volumes(vs)

    docker_client = docker.from_env()
    cmd = "/workdir/scripts/run_jupyter.sh"
    envs = docker_utils.create_envs()
    volumes = create_volumes()
    ports = docker_utils.create_ports(port)
    docker_client.containers.run(image='xihelm/vegai:{}'.format(IMAGE_TAG),
                                 name='debug',
                                 privileged=True,
                                 command=cmd,
                                 runtime='nvidia',
                                 detach=True,
                                 ports=ports,
                                 environment=envs,
                                 remove=True,
                                 shm_size="16G",
                                 volumes=volumes)


if __name__ == '__main__':
    fire.Fire(launch)
