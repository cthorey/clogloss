#!/usr/bin/env python3

import fire

from vegai.models.serving import VegaiServer


def spawn_server(host='*', port=5555):
    server = VegaiServer(host=host, port=port)
    server.start()


if __name__ == '__main__':
    fire.Fire(spawn_server)
