import boto3
import pandas as pd
import os
import joblib
import pyarrow.parquet as pq
#import category_encoders as ce
from sklearn.model_selection import train_test_split
from io import BytesIO

def save_pickle(obj, bucketname, filepath, client):
    #with open(filename, "wb") as f:
    #    pickle.dump(obj, f)
    buffer = BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    client.upload_fileobj(buffer, bucketname, filepath)