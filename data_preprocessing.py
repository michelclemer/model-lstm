# data_preprocessing.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def download_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Faz o download dos dados históricos de ações.
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df.dropna()
    return df

def preprocess_data(df: pd.DataFrame) -> (np.ndarray, MinMaxScaler):
    """
    Pré-processa os dados: seleciona a coluna 'Close' e normaliza os valores.
    """
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
