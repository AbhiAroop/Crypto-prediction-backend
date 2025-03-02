import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class PredictionModel:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = self.create_model()
        self.scaler = None

    def create_model(self):
        # Enhanced model for longer predictions
        model = keras.Sequential([
            layers.LSTM(100, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            layers.Dropout(0.2),  # Add dropout to prevent overfitting
            layers.LSTM(100, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, x_train, y_train, epochs=100, batch_size=32):
        # Increased epochs for better learning
        return self.model.fit(
            x_train, 
            y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,  # Increased validation split
            verbose=0
        )

    def predict(self, x_input, scaler):
        """Make predictions and inverse transform them to original scale"""
        predictions = self.model.predict(x_input)
        return scaler.inverse_transform(predictions)