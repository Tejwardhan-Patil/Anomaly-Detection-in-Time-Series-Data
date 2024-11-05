import unittest
import json
from app import app 

class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()  # Create a test client for the app

    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue('uptime' in data)

    def test_health_check_response_time(self):
        """Test the response time of health check endpoint"""
        import time
        start_time = time.time()
        response = self.client.get('/health')
        end_time = time.time()
        self.assertLess(end_time - start_time, 0.5)  # Check if response time is less than 500ms

    def test_predict(self):
        """Test the prediction endpoint with valid time-series data"""
        sample_data = {
            "time_series": [
                {"timestamp": "2024-06-04T00:00:00Z", "value": 100},
                {"timestamp": "2024-06-04T01:00:00Z", "value": 105},
                {"timestamp": "2024-06-04T02:00:00Z", "value": 98},
                {"timestamp": "2024-06-04T03:00:00Z", "value": 102}
            ]
        }
        response = self.client.post('/predict', data=json.dumps(sample_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], list)
        self.assertEqual(len(data['predictions']), 4)

    def test_predict_large_data(self):
        """Test the prediction endpoint with a large amount of time-series data"""
        sample_data = {
            "time_series": [{"timestamp": f"2024-06-04T{str(i).zfill(2)}:00:00Z", "value": i} for i in range(1000)]
        }
        response = self.client.post('/predict', data=json.dumps(sample_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predictions', data)
        self.assertEqual(len(data['predictions']), 1000)

    def test_predict_with_missing_values(self):
        """Test the prediction endpoint with missing values in time-series data"""
        sample_data = {
            "time_series": [
                {"timestamp": "2024-06-04T00:00:00Z", "value": 100},
                {"timestamp": "2024-06-04T01:00:00Z", "value": None},  # Missing value
                {"timestamp": "2024-06-04T02:00:00Z", "value": 98},
                {"timestamp": "2024-06-04T03:00:00Z", "value": 102}
            ]
        }
        response = self.client.post('/predict', data=json.dumps(sample_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], list)
        self.assertIsNone(data['predictions'][1])  # Ensure the missing value is handled properly

    def test_predict_invalid_data(self):
        """Test the prediction endpoint with invalid data format"""
        invalid_data = {
            "time_series": [
                {"timestamp": "2024-06-04T00:00:00Z", "value": "invalid_value"}  # Invalid value type
            ]
        }
        response = self.client.post('/predict', data=json.dumps(invalid_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "Invalid data format")

    def test_predict_empty_data(self):
        """Test the prediction endpoint with empty time-series data"""
        empty_data = {"time_series": []}
        response = self.client.post('/predict', data=json.dumps(empty_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "No time-series data provided")

    def test_model_info(self):
        """Test the endpoint providing model details"""
        response = self.client.get('/model_info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('model_name', data)
        self.assertIn('version', data)
        self.assertIn('description', data)
        self.assertEqual(data['model_name'], 'LSTM Anomaly Detector')

    def test_model_info_no_model(self):
        """Test the model info endpoint when no model is loaded"""
        # Simulating a scenario where the model is not loaded
        response = self.client.get('/model_info?model_loaded=false')
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "No model loaded")

    def test_health_check_with_dependencies(self):
        """Test the health check with external service dependencies"""
        response = self.client.get('/health?check_dependencies=true')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('db_status', data)
        self.assertIn('cache_status', data)
        self.assertEqual(data['db_status'], 'connected')
        self.assertEqual(data['cache_status'], 'healthy')

    def test_timeout_handling(self):
        """Test how the API handles timeouts on prediction requests"""
        import time
        sample_data = {
            "time_series": [
                {"timestamp": "2024-06-04T00:00:00Z", "value": 100}
            ]
        }
        time.sleep(2)  # Simulate a delay that would trigger a timeout
        response = self.client.post('/predict', data=json.dumps(sample_data), content_type='application/json')
        self.assertNotEqual(response.status_code, 408)  # Ensure API does not timeout unnecessarily

    def test_post_large_payload(self):
        """Test handling of excessively large payloads for prediction"""
        large_payload = {
            "time_series": [{"timestamp": f"2024-06-04T{str(i).zfill(2)}:00:00Z", "value": i} for i in range(10000)]
        }
        response = self.client.post('/predict', data=json.dumps(large_payload), content_type='application/json')
        self.assertEqual(response.status_code, 413)  # Payload too large error
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "Payload too large")

    def test_cors_policy(self):
        """Test CORS policy headers"""
        response = self.client.options('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')

    def test_invalid_route(self):
        """Test handling of requests to invalid routes"""
        response = self.client.get('/invalid_route')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Not Found')

    def test_model_training_status(self):
        """Test the endpoint providing model training status"""
        response = self.client.get('/training_status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'training_complete')

    def test_predict_batch(self):
        """Test the batch prediction endpoint with multiple time-series"""
        batch_data = {
            "batch": [
                {
                    "id": "series_1",
                    "time_series": [
                        {"timestamp": "2024-06-04T00:00:00Z", "value": 100},
                        {"timestamp": "2024-06-04T01:00:00Z", "value": 105}
                    ]
                },
                {
                    "id": "series_2",
                    "time_series": [
                        {"timestamp": "2024-06-04T00:00:00Z", "value": 110},
                        {"timestamp": "2024-06-04T01:00:00Z", "value": 108}
                    ]
                }
            ]
        }
        response = self.client.post('/predict_batch', data=json.dumps(batch_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('batch_predictions', data)
        self.assertEqual(len(data['batch_predictions']), 2)

if __name__ == '__main__':
    unittest.main()