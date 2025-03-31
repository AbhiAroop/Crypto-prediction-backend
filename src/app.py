from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import gc
from services.crypto_service import get_prediction_data
from models.prediction_model import PredictionModel

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://crypto-prediction-frontend.vercel.app",
            "http://localhost:3000"  # For local development
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# Initialize model and cache
model = PredictionModel()
prediction_cache = {}
MAX_CACHE_SIZE = 100

def validate_data(data):
    """Validate data format and content"""
    if isinstance(data, pd.DataFrame):
        return not data.empty
    elif isinstance(data, pd.Series):
        return not data.empty
    return False

def process_chunk(coin, chunk_days, offset):
    """Process a chunk of prediction data"""
    try:
        # Get historical data from crypto service
        data = get_prediction_data(coin)
        
        # Validate data
        if not validate_data(data):
            raise ValueError(f"Invalid or empty data received for {coin}")
            
        # Ensure data is in correct format
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
            
        # Process the prediction using your model
        result = model.predict(data, chunk_days)
        
        # Ensure result is list format
        if isinstance(result, np.ndarray):
            result = result.tolist()
        
        # Adjust timestamps based on offset
        if offset > 0:
            result = [
                [timestamp + (offset * 86400000), value] 
                for timestamp, value in result
            ]
                
        return result
        
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        raise

@app.after_request
def after_request(response):
    """Add headers to every response."""
    response.headers.add('Access-Control-Allow-Origin', 'https://crypto-prediction-frontend.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    return response

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.json
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            coin = data.get('coin')
            try:
                days = int(data.get('days', 1))
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid days parameter'}), 400
            
            if not coin or days < 1:
                return jsonify({'error': 'Missing or invalid parameters'}), 400
                
            # Memory optimization
            if len(prediction_cache) > MAX_CACHE_SIZE:
                oldest_key = min(prediction_cache.items(), key=lambda x: x[1]['timestamp'])[0]
                del prediction_cache[oldest_key]
            
            predictions = []
            chunk_size = min(50, days)  # Process data in smaller chunks
            
            try:
                for i in range(0, days, chunk_size):
                    chunk_days = min(chunk_size, days - i)
                    chunk_predictions = process_chunk(coin, chunk_days, i)
                    if chunk_predictions:
                        predictions.extend(chunk_predictions)
                    gc.collect()
                
                if not predictions:
                    raise ValueError("No predictions generated")
                
                prediction_cache[coin] = {
                    'predictions': predictions,
                    'timestamp': datetime.now(),
                    'days': days
                }
                
                return jsonify({
                    'predictions': predictions,
                    'coin': coin,
                    'days': days
                }), 200
                
            except Exception as e:
                print(f"Error processing prediction: {str(e)}")
                return jsonify({'error': f'Error processing prediction: {str(e)}'}), 500
                
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)