import os
import uuid

import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pylab as plt
from pytorch_lightning import callbacks
from sklearn import metrics
from vegai.utils import s3


def save_and_push_to_s3(payload, payload_type, default_uri):
    """
    Small util to push and get readable link for trains. For some reason,
    their stuff is not using the correct link
    """
    bucket = s3.S3Bucket(bucket_name='vegai-trains', region_name='eu-west-2')
    if not os.path.isdir('/tmp'):
        os.makedirs('/tmp')
    fpath = '/tmp/figure.png'
    if payload_type == 'fig':
        payload.savefig(fpath)
    elif payload_type == 'image':
        payload.save(fpath)
    else:
        raise NotImplementedError
    key = os.path.join(default_uri.replace('s3://vegai-trains/', ''),
                       '{}.png'.format(uuid.uuid4().hex))
    bucket.upload_from_file(fpath, key, overwrite=True)
    url = bucket.create_predesigned_url(key, expiration=3600 * 6)
    os.remove(fpath)
    return url


def plot_confusion_matrix(cm,
                          classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


class AnalysisCallback(callbacks.Callback):
    def generate_prediction_dataframe(self, tr, pl_module):
        pl_module.freeze()
        pl_module.eval()
        pl_module.to('cuda')
        id2label = {v: k for k, v in pl_module.test_set.label2id.items()}
        data = []
        for batch in pl_module.test_dataloader():
            image, targets = batch
            with torch.no_grad():
                logits = pl_module(image.to('cuda'))
                probs = nn.Softmax(dim=1)(logits)
                preds = torch.argmax(logits, dim=1)
            probs = probs.to('cpu')
            preds = preds.to('cpu')
            for idx in range(len(targets)):
                data.append({
                    'ytrue': targets[idx].item(),
                    'ypred': preds[idx].item(),
                    'label': id2label[targets[idx].item()],
                    'pred': id2label[preds[idx].item()],
                    'confidence': probs[idx][preds[idx]].item()
                })
        return pd.DataFrame(data)

    def _report_confusion_matrix(self, df, tr, pl_module, confidence=0.6):
        """
        Report the confusion matrix
        """
        tmp = df[df.confidence > confidence]
        if len(tmp) == 0:
            return
        cm = metrics.confusion_matrix(tmp.ytrue, tmp.ypred)
        fig = plot_confusion_matrix(cm,
                                    pl_module.train_set.label2id.keys(),
                                    normalize=False)
        url = save_and_push_to_s3(
            fig, 'fig', default_uri=tr.logger.experiment.reporter._storage_uri)
        tr.logger.experiment.reporter.report_image(
            'Confusion matrix',
            '@ {} confidence'.format(confidence),
            iter=0,
            src=url)
        return {
            'mcc': metrics.matthews_corrcoef(tmp.ytrue, tmp.ypred),
            'confidence': confidence,
            'f1-score': metrics.f1_score(tmp.ytrue, tmp.ypred),
            'precision': metrics.precision_score(tmp.ytrue, tmp.ypred),
            'recall': metrics.recall_score(tmp.ytrue, tmp.ypred),
            'fraction_answered': len(tmp) / float(len(df))
        }

    def report_confusion_matrix(self, df, tr, pl_module):
        data = []
        for confidence in [0.5, 0.6, 0.7, 0.8, 0.9]:
            res = self._report_confusion_matrix(df, tr, pl_module, confidence)
            if res:
                data.append(res)
        results = pd.DataFrame(data)
        tr.logger.experiment.reporter.report_table(title='metrics',
                                                   series='metrics',
                                                   table=results,
                                                   iteration=0)

    def _report_samples(self, dataset, tr, pl_module, title, n=15):
        n = min(len(dataset), n)
        df = dataset.sample(n)
        for i, idx in enumerate(df.index):
            img = pl_module.test_set.get_images(idx)
            url = save_and_push_to_s3(
                img,
                'image',
                default_uri=tr.logger.experiment.reporter._storage_uri)
            tr.logger.experiment.reporter.report_image(title,
                                                       'Index {}'.format(idx),
                                                       iter=0,
                                                       src=url)

    def report_samples(self, preds, tr, pl_module):
        fp = preds[(preds.label == 'clear') & (preds.pred == 'stalled')]
        self._report_samples(fp, tr, pl_module, title='FN clear/stalled')
        fn = preds[(preds.label == 'clear') & (preds.pred == 'clear')]
        self._report_samples(fn, tr, pl_module, title='TN clear/clear')
        tp = preds[(preds.label == 'stalled') & (preds.pred == 'clear')]
        self._report_samples(tp, tr, pl_module, title='FP stalled/clear')
        tn = preds[(preds.label == 'stalled') & (preds.pred == 'stalled')]
        self._report_samples(tn, tr, pl_module, title='TP stalled/stalled')

    def on_train_end(self, tr, pl_module):
        """
        Compute confusion matrix.
        Display confusion matrix.
        Report metrics.
        Show image of FP.
        """
        tr.logger.experiment.reporter.setup_upload('vegai-trains')
        preds = self.generate_prediction_dataframe(tr, pl_module)
        self.report_confusion_matrix(preds, tr, pl_module)
        self.report_samples(preds, tr, pl_module)
