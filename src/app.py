from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime
import gc
from services.crypto_service import get_prediction_data
from models.prediction_model import PredictionModel

app = Flask(__name__)
CORS(app)

# Initialize model and cache
model = PredictionModel()
prediction_cache = {}
MAX_CACHE_SIZE = 100

def process_chunk(coin, chunk_days, offset):
    """Process a chunk of prediction data"""
    try:
        # Get historical data from crypto service
        data = get_prediction_data(coin)  # Remove chunk_days parameter
        
        if not data or len(data) == 0:
            raise ValueError(f"No data received for {coin}")
            
        # Process the prediction using your model
        result = model.predict(data, chunk_days)
        
        # Adjust timestamps based on offset
        if offset > 0:
            for item in result:
                item[0] += offset * 86400000  # Add offset in milliseconds
                
        return result
        
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        raise

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.json
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            coin = data.get('coin')
            days = int(data.get('days', 1))
            
            if not coin or not days:
                return jsonify({'error': 'Missing required parameters'}), 400
                
            # Memory optimization: Clear old cache entries
            if len(prediction_cache) > MAX_CACHE_SIZE:
                oldest_key = min(prediction_cache.items(), key=lambda x: x[1]['timestamp'])[0]
                del prediction_cache[oldest_key]
            
            # Process data in chunks
            predictions = []
            chunk_size = 50  # Process data in smaller chunks
            
            try:
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
                
            except Exception as e:
                print(f"Error processing prediction: {str(e)}")
                return jsonify({'error': f'Error processing prediction: {str(e)}'}), 500
                
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)