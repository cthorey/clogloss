import os

from box import Box

import torch
import torch.nn as nn
from PIL import Image
from vegai.models import common
from vegai.models.clog_loss.crazy_alzheimer import config, data, mosaic
from vegai.utils import dictionary, s3


def get_cfg(config_name: str = ""):
    default_cfg = config.get().to_dict()
    flatten_default_cfg = dictionary.flatten(default_cfg)
    cfg = common.get_cfg(model_task='clog_loss',
                         model_name='crazy_alzheimer',
                         config_name=config_name)
    flatten_cfg = dictionary.flatten(cfg.to_dict())
    flatten_default_cfg.update(flatten_cfg)
    return Box(dictionary.unflatten(flatten_default_cfg))


class Model(common.BaseModel):
    """
    Instance segmentation based on mask-rcnn.
    """
    def __init__(self,
                 model_type='models',
                 model_name='crazy_alzheimer',
                 model_task='clog_loss',
                 model_description="",
                 expname=None):
        super(Model, self).__init__(model_type=model_type,
                                    model_name=model_name,
                                    model_task=model_task,
                                    model_description=model_description,
                                    expname=expname)
        cfg = config.get().to_dict()
        self.build_network(cfg, expname=expname)

    def build_network(self, cfg, expname=None):
        self.network = mosaic.MosaicNetwork(cfg, is_train=expname is None)
        if self.expname is not None:
            checkpoint_path = os.path.join(
                self.model_folder, '{}_weights.pth'.format(self.expname))
            if not os.path.isfile(checkpoint_path):
                s3.download_from_bucket(self.bucket, checkpoint_path)
            print('Reloading from {}'.format(checkpoint_path))
            self.network = self.network.load_from_checkpoint(checkpoint_path,
                                                             is_train=False)
            self.label2id = torch.load(checkpoint_path)['extra']['label2id']
            self.id2label = {v: k for k, v in self.label2id.items()}

    def prepare_inputs(self, images, device):
        transforms = data.build_transforms(self.network.cfg,
                                           stage='validation')
        inputs = dict.fromkeys(images)
        for key in images:
            imgs = images[key]
            if key not in ['image', "footprint"]:
                raise ValueError("Wrong key fellow")
            imgs = [imgs] if not isinstance(imgs, list) else imgs
            inputs[key] = torch.stack(
                [transforms(Image.fromarray(img)).to(device) for img in imgs],
                0)
        return inputs

    def _predict(self, images, device):
        device = torch.device(device)
        self.network.to(device)
        inputs = self.prepare_inputs(images, device)
        with torch.no_grad():
            logits = self.network(inputs)
        return logits

    def predict(self, images, device='cuda'):
        """
        Args:
            images (dict): dict(footprint:np.array,image: np.array)
        """
        logits = self._predict(images, device)
        _, preds = torch.max(logits, 1)
        preds = preds.to('cpu').numpy()
        return [self.id2label[idx] for idx in preds]

    def predict_prob(self, images, device='cuda'):
        """
        images (PIL.Image): list of PIL images
        """
        logits = self._predict(images, device)
        probs = nn.Softmax(dim=1)(logits)
        probs = probs.to('cpu').numpy()
        return [{v: probs[idx, k]
                 for k, v in self.id2label.items()}
                for idx in range(len(probs))]
