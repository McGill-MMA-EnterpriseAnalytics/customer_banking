import json
import pandas as pd
from utils import *
from predict import Predict as pr
def lambda_handler(event, context):
    bucket_name = 'enterprise-data-science'
    path = 'training_files/trained_model.pickle'
    #client = boto3.client('s3')
    print('predict')
    data = json.loads(json.dumps(event))
    '''
    with open(event) as f:
        data = json.load(f)
    print(data)
    '''
    
    #data = json.loads(json.dumps(event))
    try:
        data = data['result']
    except:
        data = data
    df = pd.DataFrame(data['transactions'])
    
    predictions = pr.main_predict(df, bucket_name, path)


    #results = {'results': predictions} 
    results_dicts = [json.loads(item) for item in predictions]
    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": results_dicts
    }
    return response
