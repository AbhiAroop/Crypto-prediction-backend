import unittest
import numpy as np
from unittest.mock import patch, Mock
from tensorflow import keras
import pandas as pd
# Fix imports to use relative paths
from src.app import app
from src.models.prediction_model import PredictionModel
from src.utils.data_processor import validate_input, preprocess_data, create_sequences
from src.services.crypto_service import get_prediction_data


class PredictionModelTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.model = PredictionModel()

    def test_predict_valid_input(self):
        """Test prediction with valid input parameters"""
        # Mock the crypto service and model prediction
        with patch('src.services.crypto_service.get_prediction_data') as mock_get_data, \
            patch('src.models.prediction_model.PredictionModel.predict') as mock_predict:
            
            # Set up mock data
            mock_prices = pd.DataFrame({'price': [100, 200, 300]})
            mock_get_data.return_value = mock_prices
            
            # Mock prediction output
            mock_predict.return_value = np.array([[150.0]])
            
            # Make request
            response = self.app.post('/predict', json={
                'coin': 'bitcoin',
                'days': 7
            })
            
            # Assert response
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('predictions', data)
            self.assertIn('coin', data)
            self.assertIn('days', data)

    def test_predict_invalid_coin(self):
        """Test prediction with invalid cryptocurrency"""
        response = self.app.post('/predict', json={
            'coin': 'invalidcoin',
            'days': 7
        })
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.get_json())

    def test_predict_invalid_days(self):
        """Test prediction with invalid number of days"""
        response = self.app.post('/predict', json={
            'coin': 'bitcoin',
            'days': 10
        })
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.get_json())

    def test_data_processor(self):
        """Test data processing functions"""
        # Test data preprocessing
        mock_data = {'prices': np.array([[1], [2], [3], [4], [5]])}
        normalized_data, scaler = preprocess_data(mock_data)
        self.assertEqual(normalized_data.shape[1], 1)
        self.assertTrue(0 <= normalized_data.all() <= 1)

        # Test sequence creation
        seq_length = 2
        X, y = create_sequences(normalized_data, seq_length)
        self.assertEqual(X.shape[1], seq_length)
        self.assertEqual(y.shape[0], len(normalized_data) - seq_length)

    def test_model_creation(self):
        """Test model architecture creation"""
        model = self.model.create_model()
        self.assertIsInstance(model, keras.Sequential)
        self.assertEqual(len(model.layers), 4)  # 2 LSTM + 2 Dense layers

    def test_input_validation(self):
        """Test input validation function"""
        # Valid input
        try:
            validate_input('bitcoin', 5)
        except ValueError:
            self.fail("validate_input raised ValueError unexpectedly!")

        # Invalid coin type
        with self.assertRaises(ValueError):
            validate_input(123, 5)

        # Invalid days range
        with self.assertRaises(ValueError):
            validate_input('bitcoin', 8)
        with self.assertRaises(ValueError):
            validate_input('bitcoin', 0)

    @patch('src.models.prediction_model.PredictionModel.predict')
    def test_model_prediction(self, mock_predict):
        """Test model prediction"""
        mock_input = np.random.rand(1, 10, 1)
        mock_scaler = Mock()
        mock_predict.return_value = np.array([[100.0]])

        predictions = self.model.predict(mock_input, mock_scaler)
        self.assertEqual(predictions.shape, (1, 1))

    @patch('src.services.crypto_service.requests.get')
    def test_crypto_service(self, mock_get):
        """Test cryptocurrency data fetching service"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'prices': [[1623456789000, 35000], [1623456790000, 36000]]
        }
        mock_get.return_value = mock_response

        df = get_prediction_data('bitcoin')
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

if __name__ == '__main__':
    unittest.main()