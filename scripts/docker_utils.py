import os
import uuid

from box import Box


def create_volumes(volumes):
    vs = Box()
    for v in volumes:
        from_path, bind, mode = v.split(':')
        from_path = from_path.replace('~', os.environ['HOME'])
        vs.update(Box({from_path: {'bind': bind, 'mode': mode}}))
    return vs


def create_policy(name="on-failure", retrycnt=10):
    policy = Box()
    policy.Name = name
    if name == "on-failure":
        policy.MaximumRetryCount = retrycnt
    return policy.to_dict()


def create_envs(nvidia_device=0):
    env = Box()
    env.DISPLAY = os.environ.get('DISPLAY', ':0')
    env.ROOT_DIR = "/workdir"
    env.NVIDIA_VISIBLE_DEVICES = nvidia_device
    return env


def create_ports(port=8888):
    if port is None:
        port = 8888
    p = {'8888/tcp': port}
    return p


def create_devices():
    devices = []
    devices.append('/dev/ttyUSB0:/dev/ttyUSB0')
    return devices


def parse_cmd(script, fn, **kwargs):
    cmd_args = ["--{}={}".format(k, v) for k, v in kwargs.items()]
    cmd = ["python3 -u", script, fn]
    cmd = cmd + cmd_args
    return " ".join(cmd)


def generate_name(name, config_folder):
    return '{}_{}'.format(name, uuid.uuid4().hex)
