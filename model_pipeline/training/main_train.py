
from train import train_model
from utils import *
import os
import boto3
from botocore.exceptions import ClientError



def main_train():    

    bucket_name = 'enterprise-data-science'  
    path = 'training_files/'
    client = boto3.client('s3')

    model = train_model('../Dataset/bank-full.csv')
    trained_model_path = f'{path}trained_model.pickle'
    save_pickle(model, bucket_name, trained_model_path, client)
    return model

if __name__ == '__main__':
    model = main_train()
    
