import datetime
import os

import pandas as pd
import psycopg2
from orm.dionysos import Dataset, session_scope

ROOT_DIR = os.environ['ROOT_DIR']


def add_entry_to_dataset_table(entry):
    with session_scope() as sess:
        sess.query(Dataset).filter(
            Dataset.dataset_name == entry.dataset_name).delete()
        sess.add(entry)


def info_to_database(overwrite, data_cat, maintainer, description, data_name,
                     dest_folder, nb_training_samples, nb_validation_samples):
    train_key = os.path.join(dest_folder,
                             'annotations_{}.json'.format('train')).replace(
                                 '{}/'.format(ROOT_DIR), '')
    validation_key = os.path.join(
        dest_folder, 'annotations_{}.json'.format('validation')).replace(
            '{}/'.format(ROOT_DIR), '')
    entry = Dataset(
        dataset_cat=data_cat,
        maintainer=maintainer,
        description=description,
        dataset_name=data_name,
        nb_samples=nb_training_samples + nb_validation_samples,
        nb_training_samples=nb_training_samples,
        nb_validation_samples=nb_validation_samples,
        created_on=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        train_key=train_key,
        validation_key=validation_key)
    add_entry_to_dataset_table(entry)
