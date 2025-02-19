import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
from src.logger import configure_logging
import joblib

def save_scaler(scaler_model, path):
    try:
        logging.info('Salvando o modelo scaler')
        joblib.dump(scaler_model, path)
    except Exception as e:
        logging.error(f'Erro ao salvar o modelo scaler: {e}')
        raise RuntimeError(f'Erro ao salvar o modelo scaler: {e}')

def download_stock_data(stock_symbol, period='5y'):
    """
    Baixa os dados históricos da ação usando yfinance.

    Args:
        stock_symbol (str): O símbolo da ação (ex: "AAPL").
        period (str): O período dos dados a serem baixados (ex: "5y" para 5 anos).

    Returns:
        pandas.DataFrame: DataFrame com os dados históricos da ação.
    """
    try:
        logging.info('Executando o donwload dos dados históricos da ação.')
        data = yf.download(stock_symbol, period=period ,multi_level_index=False)
        data = data[['Close']]
        data.index = pd.to_datetime(data.index)
        data.to_parquet(r'stock-price-forecaster\data\raw\{stock_symbol}.parquet')
        return data
    except Exception as e:
        logging.error(f"Erro ao processar download: {e}")
        raise RuntimeError(f"Erro ao processar download: {e}")

def standardize_data(data):
    """
    Padroniza os dados usando MinMaxScaler.

    Args:
        data (pandas.DataFrame): DataFrame com os dados a serem padronizados.

    Returns:
        pandas.DataFrame: DataFrame com os dados padronizados.
    """
    try:
        logging.info('Padronizando os dados.')
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.values)
        save_scaler(scaler, path=r'stock-price-forecaster\models\Scaler_model.pkl')
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
        return data_scaled
    except Exception as e:
        logging.error(f"Erro ao padronizar os dados: {e}")
        raise RuntimeError(f"Erro ao padronizar os dados: {e}")
        
def preprocess_data(data, sequence_lenght=60):
    """
    Pré-processar os dados para uso em modelos de aprendizado de máquina, 
    particularmente no caso de previsão de séries temporais.

    Args:
        data (pandas.DataFrame): Dados a serem processados.
        sequence_lenght (int, optional):  Número de dias a serem usados para formar a 
        sequência de entrada para o modelo. Defaults to 60.
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    
    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    try:
        logging.info('Preprocessando os dados!')
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        logging.info(f"Tamanho dos dados de treino: {len(train_data)}")
        logging.info(f"Tamanho dos dados de teste: {len(test_data)}")
        
        X_train, y_train = create_sequences(train_data.values, sequence_lenght)
        X_test, y_test = create_sequences(test_data.values, sequence_lenght)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        logging.info("Formato X_train:", X_train.shape)
        logging.info("Formato y_train:", y_train.shape)
        logging.info("Formato X_test:", X_test.shape)
        logging.info("Formato y_test:", y_test.shape)
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Erro ao preprocessar os dados: {e}")
        raise RuntimeError(f"Erro ao preprocessar os dados: {e}")