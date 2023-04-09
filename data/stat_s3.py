import boto3
import pandas as pd

S3_BUCKET = "p4h-baseball"


def get_s3_client(aws_access_key_id=None, aws_secret_access_key=None, aws_profile=None):
    assert (aws_access_key_id is not None and aws_secret_access_key is not None) or (
        aws_profile is not None
    ), "creating a client requires either aws credentials or aws profile"
    if aws_profile is not None:
        session = boto3.Session(profile_name=aws_profile)
    else:
        session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    return session.client('s3')

def s3_to_df(client, s3_path, bucket=S3_BUCKET):
    obj = client.get_object(Bucket=bucket, Key=s3_path)
    return pd.read_csv(obj['Body'])

def get_year_df(client, year, set_type='full'):
    prefix = f"raw_data/season={year}/type={set_type}/all.csv"
    return s3_to_df(client, prefix)

