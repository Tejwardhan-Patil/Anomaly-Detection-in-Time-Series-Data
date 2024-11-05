import unittest
import os
import pandas as pd
from pandas.testing import assert_frame_equal
from data.scripts.preprocess import preprocess_data
from data.scripts.feature_engineering import generate_features
from data.scripts.split import split_data

class TestDataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load raw data for testing purposes
        cls.raw_data_path = 'data/raw/test_data.csv'
        cls.processed_data_path = 'data/processed/test_data_processed.csv'
        cls.features_data_path = 'data/features/test_data_features.csv'

        # Check if test data exists
        assert os.path.exists(cls.raw_data_path), "Raw data not found"
        cls.raw_data = pd.read_csv(cls.raw_data_path)

    def test_preprocess_data(self):
        # Test data preprocessing
        processed_data = preprocess_data(self.raw_data)

        # Ensure preprocessing returns non-empty data
        self.assertIsNotNone(processed_data, "Preprocessing returned None")
        self.assertFalse(processed_data.empty, "Preprocessing returned empty data")

        # Check that missing values are handled
        self.assertTrue(processed_data.isnull().sum().sum() == 0, "Processed data contains missing values")

        # Check if specific columns have been normalized or resampled correctly ('value')
        self.assertTrue('value' in processed_data.columns, "'value' column missing after preprocessing")
        self.assertTrue((processed_data['value'] >= 0).all(), "Invalid values after normalization")

        # Save processed data for further tests
        processed_data.to_csv(self.processed_data_path, index=False)

    def test_preprocess_data_shape(self):
        # Test if the number of columns after preprocessing is as expected
        processed_data = pd.read_csv(self.processed_data_path)
        expected_columns = ['timestamp', 'value'] 
        self.assertListEqual(list(processed_data.columns), expected_columns, "Columns do not match expected columns after preprocessing")

    def test_preprocess_empty_input(self):
        # Test preprocessing with an empty DataFrame
        empty_data = pd.DataFrame(columns=['timestamp', 'value'])
        processed_data = preprocess_data(empty_data)
        self.assertTrue(processed_data.empty, "Preprocessing non-empty data when it should be empty")

    def test_preprocess_invalid_input(self):
        # Test preprocessing with invalid data types
        invalid_data = pd.DataFrame({
            'timestamp': ['invalid', 'invalid'],
            'value': ['invalid', 'invalid']
        })
        with self.assertRaises(ValueError, msg="Preprocessing did not raise ValueError on invalid input"):
            preprocess_data(invalid_data)

    def test_generate_features(self):
        # Test feature generation
        processed_data = pd.read_csv(self.processed_data_path)
        features_data = generate_features(processed_data)

        # Ensure feature generation returns non-empty data
        self.assertIsNotNone(features_data, "Feature generation returned None")
        self.assertFalse(features_data.empty, "Feature generation returned empty data")

        # Check that new features are added
        self.assertGreater(features_data.shape[1], processed_data.shape[1], "No new features generated")
        
        # Ensure specific features like 'rolling_mean' are generated
        self.assertIn('rolling_mean', features_data.columns, "'rolling_mean' feature missing")

        # Save features data for further tests
        features_data.to_csv(self.features_data_path, index=False)

    def test_generate_features_type(self):
        # Test feature generation to ensure feature types are correct
        processed_data = pd.read_csv(self.processed_data_path)
        features_data = generate_features(processed_data)
        
        # Check that 'rolling_mean' is a float
        self.assertEqual(features_data['rolling_mean'].dtype, 'float64', "Feature 'rolling_mean' is not float")

    def test_generate_features_empty_input(self):
        # Test feature generation with an empty DataFrame
        empty_data = pd.DataFrame(columns=['timestamp', 'value'])
        features_data = generate_features(empty_data)
        self.assertTrue(features_data.empty, "Feature generation returned non-empty data on empty input")

    def test_split_data(self):
        # Test splitting data into training, validation, and test sets
        features_data = pd.read_csv(self.features_data_path)
        train_data, val_data, test_data = split_data(features_data)

        # Ensure splits are non-empty
        self.assertTrue(len(train_data) > 0, "Training data is empty")
        self.assertTrue(len(val_data) > 0, "Validation data is empty")
        self.assertTrue(len(test_data) > 0, "Test data is empty")

        # Ensure splits are mutually exclusive
        self.assertTrue(train_data.index.isin(val_data.index).sum() == 0, "Training and validation sets overlap")
        self.assertTrue(train_data.index.isin(test_data.index).sum() == 0, "Training and test sets overlap")
        self.assertTrue(val_data.index.isin(test_data.index).sum() == 0, "Validation and test sets overlap")

    def test_split_data_ratio(self):
        # Test splitting with a specific ratio
        features_data = pd.read_csv(self.features_data_path)
        train_data, val_data, test_data = split_data(features_data, train_ratio=0.7, val_ratio=0.2)

        # Check the ratio of splits
        total_len = len(features_data)
        self.assertAlmostEqual(len(train_data) / total_len, 0.7, delta=0.05, msg="Training set size not within expected ratio")
        self.assertAlmostEqual(len(val_data) / total_len, 0.2, delta=0.05, msg="Validation set size not within expected ratio")
        self.assertAlmostEqual(len(test_data) / total_len, 0.1, delta=0.05, msg="Test set size not within expected ratio")

    def test_split_data_empty_input(self):
        # Test splitting with an empty DataFrame
        empty_data = pd.DataFrame(columns=['timestamp', 'value', 'rolling_mean'])
        train_data, val_data, test_data = split_data(empty_data)
        self.assertTrue(train_data.empty and val_data.empty and test_data.empty, "Splitting returned non-empty data on empty input")

    def test_preprocess_and_feature_integration(self):
        # Test the entire pipeline from preprocessing to feature generation
        processed_data = preprocess_data(self.raw_data)
        features_data = generate_features(processed_data)

        # Ensure the data passes through each stage correctly
        self.assertGreater(features_data.shape[1], processed_data.shape[1], "No features generated after preprocessing")
        self.assertTrue(features_data.isnull().sum().sum() == 0, "Generated features contain missing values")

    def test_split_data_consistency(self):
        # Ensure that running split multiple times produces consistent results
        features_data = pd.read_csv(self.features_data_path)
        train_data_1, val_data_1, test_data_1 = split_data(features_data)
        train_data_2, val_data_2, test_data_2 = split_data(features_data)

        # Use assert_frame_equal for DataFrame comparison
        assert_frame_equal(train_data_1, train_data_2, "Train splits are inconsistent")
        assert_frame_equal(val_data_1, val_data_2, "Validation splits are inconsistent")
        assert_frame_equal(test_data_1, test_data_2, "Test splits are inconsistent")

if __name__ == '__main__':
    unittest.main()