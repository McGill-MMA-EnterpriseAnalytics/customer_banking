from utils import * 
import numpy
import pickle
class Predict:
    def main_predict(data, bucket_name, path):
        #model = load_models_from_s3(bucket_name,  path )
        model = pickle.load(open("09_model_pipeline/predict/src/trained_model.pickle",'rb'))
        probas = model.predict_proba(data)
        preds = np.where(probas[:,1] >= 0.4, 1, 0)
        data['prob_0'] = probas[:,0]
        data['prob_1'] = probas[:,1]
        data['predictions'] = preds
        record = data.to_dict(orient='records')[0]
        json_result = data.to_json(orient='records', lines=True).splitlines()
        return json_result
