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
            "https://crypto-prediction-frontend.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True,
        "max_age": 120
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
            print(f"Received request data: {data}")
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            coin = data.get('coin')
            print(f"Selected cryptocurrency: {coin}")
            
            if not coin:
                return jsonify({'error': 'No cryptocurrency specified'}), 400
                
            days = int(data.get('days', 1))
            print(f"Prediction days: {days}")
            
            # Validate input
            validate_input(coin, days)
            
            # Fetch and process data
            prices = get_prediction_data(coin)
            print(f"Fetched price data shape: {len(prices) if prices is not None else 'None'}")
            
            if prices is None or len(prices) == 0:
                return jsonify({'error': 'No data available for this cryptocurrency'}), 404
                
            # Preprocess data
            normalized_data, scaler = preprocess_data(prices)
            print(f"Normalized data shape: {normalized_data.shape}")
            
            # Create sequences for prediction
            X, y = create_sequences(normalized_data, model.sequence_length)
            print(f"Training data shapes - X: {X.shape}, y: {y.shape}")
            
            # Train model
            model.train(X, y)
            
            # Generate predictions
            last_sequence = normalized_data[-model.sequence_length:]
            predictions = []
            
            # Updated to handle more days
            for i in range(days):
                current_sequence = last_sequence.reshape((1, model.sequence_length, 1))
                next_pred = model.predict(current_sequence, scaler)[0][0]
                predictions.append(float(next_pred))
                print(f"Day {i+1} prediction: ${next_pred:.2f}")
                
                # Update sequence for next prediction with rolling window
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = scaler.transform([[next_pred]])[0]
                
                # Add progress logging for longer predictions
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{days} days predicted")
            
            print(f"Final predictions: {predictions}")
            
            # Store predictions in cache
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