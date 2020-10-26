import importlib
import time

import fire


def create_dataset(model_task, dataset_type, data_name, overwrite, overfit,
                   **kwargs):
    try:
        module = 'src.data.{}.{}'.format(model_task, dataset_type)
        print('Loading from module: {}'.format(module))
        # wait for ros to setup
        time.sleep(2)
        module = importlib.import_module(module)
        print('Creating dataset')
        module.create_dataset(data_name=data_name,
                              overwrite=overwrite,
                              overfit=overfit,
                              **kwargs)
    except Exception as e:
        print(e)
        time.sleep(1000000)


if __name__ == '__main__':
    fire.Fire(create_dataset)
