from __future__ import print_function
import pandas as pd
import os 
import boto3
from datetime import datetime
ts = datetime.now().strftime("%Y-%m-%d")

if 'heroku' in os.environ and os.environ['heroku']:
  s3_client = boto3.client('s3', aws_access_key_id=os.environ['S3_KEY'], aws_secret_access_key=os.environ['S3_secret'])
  s3_resource = boto3.resource('s3', aws_access_key_id=os.environ['S3_KEY'], aws_secret_access_key=os.environ['S3_secret'])
else:
  boto3.setup_default_session(profile_name='hawaii')
  s3_client = boto3.client('s3')
  s3_resource = boto3.resource('s3')

bucket_name = 'hawaii-marine-debris'
version='v3'

model_path = "/tmp/debris_model_{}_10_6.h5".format(version)
print(model_path)
model_key = "ml/model/debris_model_{}_10_6.h5".format(version)
print(model_key)
if not os.path.isfile(model_path):
    s3_resource.Bucket(bucket_name).download_file(model_key, model_path)

label_path = "/tmp/labels_{}.csv".format(version)
s3_resource.Bucket(bucket_name).download_file('ml/{}/data/labels.csv'.format(version), label_path)
label_df = pd.read_csv(label_path,names=['label','id'])
label_df
label_lookup = label_df.set_index('id').T.to_dict('records')[0]
label_lookup 
