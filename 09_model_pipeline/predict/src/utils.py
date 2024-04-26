import boto3
from io import BytesIO
import joblib
def load_models_from_s3(bucket_name, object_name):
    
    s3 = boto3.client('s3')
    model_file = BytesIO()
    s3.download_fileobj(bucket_name, object_name, model_file)

    model_file.seek(0)


    modelo = joblib.load(model_file)
    
    return modelo