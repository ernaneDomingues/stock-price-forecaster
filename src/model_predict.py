import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging
from src.logger import configure_logging

def load_model(symbol):
    """_summary_

    Args:
        symbol (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    try:
        logging.info('Carregando o modelo.')
        model = load_model(fr'stock-price-forecaster\models\{symbol}_lstm_model.keras')
        return model
    except Exception as e:
        logging.error(f'Erro ao carregar o modelo: {e}')
        raise RuntimeError(f'Erro ao carregar o modelo: {e}')
    
def predict_price(symbol, period='60d', time_steps=60):
    """_summary_

    Args:
        symbol (_type_): _description_
        period (str, optional): _description_. Defaults to '60d'.
        time_steps (int, optional): _description_. Defaults to 60.

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    try:
        logging.info('Realizando a predição.')
        data = yf.download(symbol, period=period, interval='1d', multi_level_index=False)
        data = data[['Close']]
        data.index = pd.to_datetime(data.index)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values)
        df_scaled = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        train_size = int(len(df_scaled) * 0.8)
        train_data = df_scaled[:train_size]
        test_data = df_scaled[train_size:]
        
        model = load_model(symbol)
        ultimos_dias_teste = test_data[-time_steps:].values
        input_scaled = scaler.transform(ultimos_dias_teste)
        X_input = np.reshape(input_scaled, (1, input_scaled.shape[0], 1))
        previsao_escalada = model.predict(X_input)
        previsao_desnormalizada = scaler.inverse_transform(previsao_escalada)
        return previsao_desnormalizada
    except Exception as e:
        logging.error(f'Erro realizar a predição: {e}')
        raise RuntimeError(f'Erro realizar a predição: {e}')       