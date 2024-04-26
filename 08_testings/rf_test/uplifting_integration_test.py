import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from uplifting_model.utils.functions import get_datasets, get_predictions, training_process, predict_process, get_metrics

class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        cls.data = pd.DataFrame({
            'age': np.random.randint(18, 70, size=100),
            'job': np.random.choice([0, 1], size=100),
            'marital': np.random.choice([0, 1], size=100),
            'education': np.random.choice([1, 2, 3], size=100),
            'balance': np.random.randint(-2000, 5000, size=100),
            'housing': np.random.choice([0, 1], size=100),
            'loan': np.random.choice([0, 1], size=100),
            'contact': np.random.choice([0, 1], size=100),
            'day': np.random.choice(range(1, 31), size=100),
            'month': np.random.choice(range(1, 13), size=100),
            'duration': np.random.randint(30, 600, size=100),
            'campaign': np.random.randint(1, 10, size=100),
            'pdays': np.random.randint(-1, 999, size=100),
            'previous': np.random.randint(0, 10, size=100)
        })
        cls.vars = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous']
        cls.target = 'loan'
        cls.model = RandomForestClassifier()

    def test_full_pipeline(self):
        x_train, x_val, x_test, y_train, y_val, y_test = get_datasets(self.data, self.vars, self.target) 
        
        model, y_val_pred, y_proba_val = training_process(self.model, x_train, y_train, x_val, y_val)

        y_pred, y_proba, metrics = predict_process(model, 0.5, x_test, y_test)
        
        self.assertIsNotNone(metrics, "Metrics should not be None")
        self.assertTrue('accuracy' in metrics, "Accuracy should be calculated")
        self.assertTrue('roc_auc' in metrics, "ROC AUC should be calculated")

        self.assertEqual(len(y_pred), len(y_test), "Predictions length should match test labels length")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



