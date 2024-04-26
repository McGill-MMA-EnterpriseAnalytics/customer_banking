import unittest
import pandas as pd
from train import train_model, get_variables
from preprocessing import load_data, build_pipeline, split_data
from preprocess import preprocess_data, get_data, map_target
from sklearn.pipeline import Pipeline

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # using test data
        self.df = pd.read_csv('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/testing_set.csv')

    def test_load_data(self):
        try:
            df = load_data('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/testing_set.csv')
            self.assertIsNotNone(df)
            self.assertFalse(df.empty)
            self.assertIn('y', df.columns)
        except Exception as e:
            self.fail(f"Failed due to {str(e)}")


    def test_preprocess_data(self):
        df_processed = preprocess_data('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/testing_set.csv')
        self.assertNotIn('contact', df_processed.columns)
        self.assertNotIn('poutcome', df_processed.columns)
        self.assertNotIn('duration', df_processed.columns)

    def test_build_pipeline(self):
        pipeline = build_pipeline(self.df, ['job'])
        self.assertIsInstance(pipeline, Pipeline)

class TestDataManipulations(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.read_csv('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/testing_set.csv')

    def test_map_target(self):
        df_mapped = map_target(self.df)
        self.assertTrue((df_mapped['y'] == 1).any())

class TestModelTraining(unittest.TestCase):

    def test_train_model(self):
        model = train_model('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/testing_set.csv')
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

