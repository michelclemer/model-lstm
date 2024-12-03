# model_training.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pickle

def build_model(input_shape):
    """
    Constrói o modelo LSTM.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, Y_train, X_test, Y_test, epochs=20, batch_size=32):
    """
    Treina o modelo com os dados de treinamento.
    """
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test)
    )
    return history

def save_model(model, scaler, model_path='lstm_stock_model.h5', scaler_path='scaler.pkl'):
    """
    Salva o modelo treinado e o scaler.
    """
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
