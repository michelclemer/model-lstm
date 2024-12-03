# utils.py

import numpy as np

def create_dataset(dataset: np.ndarray, look_back: int) -> (np.ndarray, np.ndarray):
    """
    Cria conjuntos de dados para treinamento e teste com base no look_back.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def split_data(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Divide os dados em conjuntos de treinamento e teste sem embaralhar.
    """
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    Y_train = Y[:split_index]
    X_test = X[split_index:]
    Y_test = Y[split_index:]
    return X_train, X_test, Y_train, Y_test
