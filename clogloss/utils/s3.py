import os
import warnings
from glob import glob

import boto3
import botocore
from boto3.session import Session
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']


def download_from_bucket(bucket, local_path):
    key = local_path.replace('{}/'.format(ROOT_DIR), '')
    if not bucket.ping_key(key):
        raise RuntimeError('Could not find {} on s3'.format(key))
    if not os.path.isdir(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    bucket.download(key, local_path)


class S3Bucket(object):
    """
    Class to interact with s3
        You should have a config file in ~/.aws/credentials that reads:
        [default]
        aws_access_key_id = *********
        aws_secret_access_key = ************

    """
    def __init__(self, bucket_name=None, region_name=None):
        self.session = Session(region_name=region_name)
        self.client = boto3.client('s3',
                                   config=boto3.session.Config(
                                       signature_version='s3v4',
                                       region_name=region_name))
        self.s3service = self.session.resource('s3')
        self.bucket = self.s3service.Bucket(bucket_name)
        self.bucket_name = bucket_name

    def get_url(self, key):
        location = self.client.get_bucket_location(
            Bucket=self.bucket_name)['LocationConstraint']
        url = "https://s3-%s.amazonaws.com/%s/%s" % (location,
                                                     self.bucket_name, key)
        return url

    def create_predesigned_url(self, object_name, expiration):
        """
        Args:
            object_name (str)
            expiration (int): Time in seconds for the presigned URL to remain valid

        Returns:
            url
        """
        try:
            response = self.client.generate_presigned_url('get_object',
                                                          Params={
                                                              'Bucket':
                                                              self.bucket_name,
                                                              'Key':
                                                              object_name
                                                          },
                                                          ExpiresIn=expiration)
        except ClientError as e:
            response = None
        return response

    def download_folder(self, folder_key):
        """
        Download the content of a folder locally
        """
        folder_path = os.path.join(ROOT_DIR, folder_key)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        for obj_key in self.get_matching_s3_keys(prefix=str(folder_key)):
            fname = os.path.join(ROOT_DIR, obj_key)
            self.download(obj_key, fname)

    def get_matching_s3_keys(self, prefix='', suffix=''):
        """
        Generate the keys in an S3 bucket.
        :param bucket: Name of the S3 bucket.
        :param prefix: Only fetch keys that start with this prefix (optional).
        :param suffix: Only fetch keys that end with this suffix (optional).

        Shamelessly stolen from https://alexwlchan.net/2017/07/listing-s3-keys/
        """

        kwargs = {'Bucket': self.bucket_name}
        # If the prefix is a single string (not a tuple of strings), we can
        # do the filtering directly in the S3 API.
        if isinstance(prefix, str) or isinstance(prefix, unicode):
            kwargs['Prefix'] = str(prefix)
        ite = 0
        while True:

            # The S3 API response is a large blob of metadata.
            # 'Contents' contains information about the listed objects.
            try:
                resp = self.client.list_objects_v2(**kwargs)
            except botocore.exceptions.EndpointConnectionError:
                break
            if 'Contents' not in resp:
                break
            for obj in resp['Contents']:
                key = obj['Key']
                if key.startswith(prefix) and key.endswith(suffix):
                    yield key

            # The S3 API is paginated, returning up to 1000 keys at a time.
            # Pass the continuation token into the next response, until we
            # reach the final page (when this field is missing).
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break
            ite += 1
            if ite % 5 == 0:
                print('Quiet a lot of stuff here - looking on the next page.')

    def list_folders(self, prefix='', delimiter='/'):
        """
        List keys given a specific prefix
        """
        warnings.warn(
            'DEPRECTATED - use get_matching_s3_key instead which handle pagination'
        )
        filter_bucket = self.bucket.meta.client.list_objects(
            Bucket=self.bucket.name, Prefix=prefix, Delimiter=delimiter)
        if 'CommonPrefixes' not in filter_bucket.keys():
            return []
        data = [d for d in filter_bucket['CommonPrefixes']]
        if data:
            return [data['Prefix'] for data in data]
        else:
            return []

    def download(self, key, fname):
        """
        Download a specific file from the s3 bucket
        Args
            key (str)
            fname (str): path where download is written
        """
        self.bucket.download_file(key, fname)

    def get(self, key):
        """
        Get an obhect stored at ta ey
        """
        obj = self.s3service.Object(bucket_name=self.bucket_name, key=key)
        return obj.get()['Body']

    def upload_folder(self, folder_path, overwrite):
        """
        Upload an entire folder to s3
        """

        files = [
            f1 for f2 in os.walk(folder_path)
            for f1 in glob(os.path.join(f2[0], '*')) if os.path.isfile(f1)
        ]
        for fname in tqdm(files):
            key = fname.replace('{}/'.format(ROOT_DIR), '')
            self.upload_from_file(fname, key, overwrite)

    def ping_folder_key(self, folder_key):
        pings = self.list_folders(folder_key)
        if pings:
            return True
        else:
            return False

    def ping_key(self, key):
        """
        Checks whether file is written at this key in S3 bucket resource
        Arg
            key (str)
        Returns
            bool
        """
        try:
            self.s3service.Object(bucket_name=self.bucket_name, key=key).load()
            return True
        except botocore.exceptions.ClientError as exception:
            if exception.response['Error']['Code'] == "404":
                return False
            else:
                raise

    def upload_from_file(self, fname, key, overwrite=False):
        """
        Upload a written file to S3
        Args
            fname (str): filename to upload
            key (str): full key filename under which the file will be written
        """
        if not overwrite and self.ping_key(key):
            raise KeyError('Key exists, set overwrite to True.')
        self.bucket.upload_file(fname, key)
        return

    def insert(self, key, obj, overwrite=False):
        """
        Inserts an object to the given key, only if that object doesn't exist
        """
        if self.ping_key(key) and not overwrite:
            print('Key {} already exists'.format(key))
        else:
            self.bucket.put_object(Key=key, Body=obj)
