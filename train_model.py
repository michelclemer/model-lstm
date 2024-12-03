# train_model.py

from data_preprocessing import download_data, preprocess_data
from utils import create_dataset, split_data
from model_training import build_model, train_model, save_model
import numpy as np

def main():
    # Parâmetros
    symbol = 'AAPL'  # Símbolo da ação
    start_date = '2018-01-01'
    end_date = '2024-07-20'
    look_back = 60

    # 1. Coleta de dados
    print("Baixando os dados...")
    df = download_data(symbol, start_date, end_date)

    # 2. Pré-processamento
    print("Pré-processando os dados...")
    scaled_data, scaler = preprocess_data(df)

    # 3. Criação dos conjuntos de dados
    print("Criando conjuntos de dados...")
    X, Y = create_dataset(scaled_data, look_back)

    # 4. Divisão em treino e teste
    print("Dividindo os dados em treinamento e teste...")
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)

    # 5. Redimensionar os dados para LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 6. Construção e treinamento do modelo
    print("Construindo e treinando o modelo...")
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)
    history = train_model(model, X_train, Y_train, X_test, Y_test, epochs=20, batch_size=32)

    # 7. Salvar o modelo e o scaler
    print("Salvando o modelo e o scaler...")
    save_model(model, scaler)

    print("Treinamento e salvamento do modelo concluídos.")

if __name__ == "__main__":
    main()
