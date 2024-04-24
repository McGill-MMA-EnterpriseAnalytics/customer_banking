import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils.functions import get_datasets, get_predictions, training_process, predict_process, get_metrics, plot_roc_curve, plot_precision_recall

class TestProjectFunctions(unittest.TestCase):

    def setUp(self):
        # sample data for testing purpose
        data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'job': [1, 0, 1, 0],
            'marital': [0, 1, 0, 1],
            'education': [1, 2, 1, 3],
            'balance': [100, 150, 200, 250],
            'housing': [1, 1, 0, 0],
            'loan': [0, 0, 1, 1],
            'contact': [1, 0, 1, 0],
            'day': [5, 10, 15, 20],
            'month': [1, 2, 1, 2],
            'duration': [100, 200, 300, 400],
            'campaign': [1, 1, 2, 2],
            'pdays': [0, -1, 0, -1],
            'previous': [0, 1, 0, 1]
        })
        self.vars = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous']
        self.target = 'loan'
        self.model = RandomForestClassifier()
        self.data = data

    def test_predict_process(self):
        x_train, x_val, x_test, y_train, y_val, y_test = get_datasets(self.data, self.vars, self.target, stratify=self.data[self.target])
        self.model.fit(x_train, y_train)
        y_pred, y_proba, metrics = predict_process(self.model, 0.5, x_test, y_test)
        self.assertIsNotNone(metrics)
        self.assertTrue('roc_auc' in metrics)

    def test_training_process(self):
        x_train, x_val, x_test, y_train, y_val, y_test = get_datasets(self.data, self.vars, self.target, stratify=self.data[self.target])
        self.model.fit(x_train, y_train)
        model, y_val_pred, y_proba_val = training_process(self.model, x_train, y_train, x_val, y_val)
        self.assertIsNotNone(model)

    def get_metrics(y_test, y_pred, y_proba):
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        if len(np.unique(y_test)) > 1:  # Check if there are at least two classes
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:,1])
        else:
            metrics['roc_auc'] = float('nan')  # Not defined if only one class
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        return metrics


    def test_get_predictions(self):
        x_train, x_val, x_test, y_train, y_val, y_test = get_datasets(self.data, self.vars, self.target)
        self.model.fit(x_train, y_train)
        y_pred, y_proba = get_predictions(self.model, 0.5, x_test)
        # checking predictions and probabilities are not None
        self.assertIsNotNone(y_pred)
        self.assertIsNotNone(y_proba)

        # checking predictions and probabilities types
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsInstance(y_proba, np.ndarray)

    def training_process(model, x_train, y_train, x_val, y_val):
        model.fit(x_train, y_train) 
        y_val_pred = model.predict(x_val)  
        y_proba_val = model.predict_proba(x_val)[:, 1]  
        return model, y_val_pred, y_proba_val



    def test_predict_process(self):
        x_train, x_val, x_test, y_train, y_val, y_test = get_datasets(self.data, self.vars, self.target)
        self.model.fit(x_train, y_train)
        y_pred, y_proba, metrics = predict_process(self.model, 0.5, x_test, y_test)
        # checking predictions, probabilities, and metrics
        self.assertIsNotNone(y_pred)
        self.assertIsNotNone(y_proba)
        self.assertIsNotNone(metrics)

        # checking metrics contains essential keys
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('precision' in metrics)
        self.assertTrue('recall' in metrics)
        self.assertTrue('f1' in metrics)
        self.assertTrue('roc_auc' in metrics)
        self.assertTrue('confusion_matrix' in metrics)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

