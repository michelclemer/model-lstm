# model_prediction.py

from tensorflow.keras.models import load_model
import numpy as np
import pickle

def load_trained_model(model_path='lstm_stock_model.h5', scaler_path='scaler.pkl'):
    """
    Carrega o modelo treinado e o scaler.
    """
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_price(model, scaler, input_data):
    """
    Faz a previsão do preço com base nos dados de entrada.
    """
    input_data = np.array(input_data).reshape(-1, 1)
    input_data = scaler.transform(input_data)
    input_data = np.reshape(input_data, (1, input_data.shape[0], 1))
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction)
    return float(prediction[0][0])
