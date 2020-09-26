"""
script to download all data from s3 folder. you will need to do 'aws configure' on host before run. 
If you don't have it, then install using:

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

"""

import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
import boto3
from boto import boto
import re
import ssl
import pandas as pd
import numpy as np
from configparser import ConfigParser
import json
import pickle

import configparser
import os.path
from os import path
from importlib import reload


creds_path_ar = ["../../credentials.ini","credentials.colab.ini","credentials.ini"]

for creds_path in creds_path_ar:
    if path.exists(creds_path):
        config_parser = configparser.ConfigParser()
        config_parser.read(creds_path)
        PATH_ROOT = config_parser['MAIN']['PATH_ROOT']
        PATH_DATA = config_parser['MAIN']['PATH_DATA']
        break

cred = boto3.Session().get_credentials()
ACCESS_KEY = cred.access_key
SECRET_KEY = cred.secret_key

BUCKET='sota-mafat'

conn = S3Connection(ACCESS_KEY, SECRET_KEY)
conn.auth_region_name = 'us-east-1.amazonaws.com'
mybucket = conn.get_bucket(BUCKET)

for key_name in mybucket.list():
    if (".csv" in str(key_name)) or (".pkl" in str(key_name)):
        key = mybucket.get_key(key_name.key)
        print(f"getting:{key_name.name}")
        key.get_contents_to_filename(f"{PATH_DATA}/{key_name.name}")

