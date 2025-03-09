from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from datetime import datetime
from services.crypto_service import get_prediction_data
from models.prediction_model import PredictionModel
from utils.data_processor import preprocess_data, create_sequences, validate_input

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://crypto-prediction-frontend-e0brht1kb-abhiram-aroops-projects.vercel.app",
            "https://crypto-prediction-frontend.vercel.app",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://crypto-prediction-frontend.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Initialize the prediction model
model = PredictionModel()
prediction_cache = {}
MAX_CACHE_SIZE = 100  # Limit cache size to avoid memory issues

# Store predictions in memory (consider using a proper database for production)
prediction_cache = {}

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Crypto Prediction API is running"}), 200

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.json
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            coin = data.get('coin')
            days = int(data.get('days', 1))
            
            # Memory optimization: Clear old cache entries
            if len(prediction_cache) > MAX_CACHE_SIZE:
                oldest_key = min(prediction_cache.items(), key=lambda x: x[1]['timestamp'])[0]
                del prediction_cache[oldest_key]
            
            # Process data in chunks
            predictions = []
            chunk_size = 50  # Process data in smaller chunks
            
            for i in range(0, days, chunk_size):
                chunk_days = min(chunk_size, days - i)
                chunk_predictions = process_chunk(coin, chunk_days, i)
                predictions.extend(chunk_predictions)
                gc.collect()  # Force garbage collection
            
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
            
        elif request.method == 'GET':
            try:
                coin = request.args.get('coin')
                print(f"GET request for coin: {coin}")
                
                if not coin:
                    return jsonify({'error': 'No cryptocurrency specified'}), 400
                    
                if coin in prediction_cache:
                    cached_data = prediction_cache[coin]
                    cache_age = (datetime.now() - cached_data['timestamp']).seconds
                    print(f"Found cached data for {coin}, age: {cache_age} seconds")
                    
                    # Return cached predictions if they're less than 5 minutes old
                    if cache_age < 300:
                        print(f"Returning cached predictions: {cached_data['predictions']}")
                        return jsonify({
                            'predictions': cached_data['predictions'],
                            'coin': coin,
                            'days': cached_data['days'],
                            'cached': True
                        }), 200
                
                print(f"No valid cached data found for {coin}")
                return jsonify({'error': 'No predictions available'}), 404
                
            except Exception as e:
                print(f"Error in GET request: {str(e)}")
                return jsonify({'error': f'Error fetching cached predictions: {str(e)}'}), 500
                
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)