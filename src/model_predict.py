import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging
from src.logger import configure_logging
import joblib
from src.data_handler import download_stock_data

# Variáveis globais para armazenar o modelo e o scaler carregados (serão inicializadas na inicialização da API)
MODEL = None
SCALER = None

def load_model_for_api(model_dir='models'):
    """Carrega o modelo LSTM para uso na API, armazenando-o em variável global.

    Args:
        model_dir (str, optional): Diretório onde o modelo está salvo. Padrão é 'models'.

    Raises:
        RuntimeError: Se ocorrer um erro ao carregar o modelo.
    """
    global MODEL # Indica que estamos usando a variável global MODEL
    try:
        logging.info(f'Carregando o modelo para API do diretório: {model_dir}')
        model_path = os.path.join(model_dir, 'lstm_model.keras')
        MODEL = load_model(model_path) # Carrega o modelo e armazena na variável global
        logging.info(f'Modelo para API carregado com sucesso de: {model_path}')
    except Exception as e:
        logging.error(f'Erro ao carregar o modelo para API de {model_dir}: {e}')
        raise RuntimeError(f'Erro ao carregar o modelo para API: {e}')

def load_scaler_for_api(model_dir='models'):
    """Carrega o scaler MinMaxScaler para uso na API, armazenando-o em variável global.

    Args:
        model_dir (str, optional): Diretório onde o scaler está salvo. Padrão é 'models'.

    Raises:
        RuntimeError: Se ocorrer um erro ao carregar o scaler.
    """
    global SCALER # Indica que estamos usando a variável global SCALER
    try:
        logging.info(f'Carregando o scaler para API do diretório: {model_dir}')
        scaler_path = os.path.join(model_dir, 'Scaler_model.pkl')
        SCALER = joblib.load(scaler_path) # Carrega o scaler e armazena na variável global
        logging.info(f'Scaler para API carregado com sucesso de: {scaler_path}')
    except Exception as e:
        logging.error(f'Erro ao carregar o scaler para API de {model_dir}: {e}')
        raise RuntimeError(f'Erro ao carregar o scaler para API: {e}')


def predict_price_for_api(symbol, start_date, end_date, time_steps=60):
    """Realiza a predição do preço de fechamento da ação para o período especificado,
       usando o modelo e scaler já carregados globalmente.

    Args:
        symbol (str): Símbolo da ação (ex: AAPL).
        start_date (str): Data de início para baixar os dados (YYYY-MM-DD).
        end_date (str): Data de fim para baixar os dados (YYYY-MM-DD).
        time_steps (int, optional): Tamanho da janela de tempo (sequência) usada pelo modelo LSTM. Padrão é 60.

    Raises:
        RuntimeError: Se o modelo ou scaler não estiverem carregados, ou se ocorrer outro erro durante a predição.

    Returns:
        dict: Dicionário contendo o preço de fechamento previsto (desnormalizado). Ex: {'predicted_price': 123.45}.
              Retorna None em caso de falha na predição ou dados insuficientes.
    """
    global MODEL, SCALER # Indica que estamos usando as variáveis globais MODEL e SCALER

    if MODEL is None or SCALER is None: # Verificação se o modelo e scaler foram carregados
        logging.error('Modelo ou Scaler não foram carregados. Verifique a inicialização da API.')
        raise RuntimeError('Modelo ou Scaler não inicializados para predição.')

    try:
        logging.info(f'Realizando a predição para {symbol} de {start_date} até {end_date} (API).')

        # Usando as datas de entrada fornecidas para baixar os dados
        logging.info(f'Baixando dados de {symbol} de {start_date} até {end_date} para predição (API).')
        data = download_stock_data(symbol, start_date=start_date, end_date=end_date)

        if data.empty: # Verificação importante se não houver dados baixados
            logging.warning(f"Não foram encontrados dados para {symbol} no período {start_date} - {end_date}. Impossível realizar a predição (API).")
            return None # Retornar None em caso de dados vazios

        scaled_data = SCALER.transform(data.values) # Usando o SCALER global
        df_scaled = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        # Preparar os dados de entrada para predição usando os últimos 'time_steps' dias dos dados baixados
        ultimos_dias_scaled = df_scaled[-time_steps:].values
        if ultimos_dias_scaled.shape[0] < time_steps: # Verificação importante: se não houver dados suficientes
            logging.warning(f"Não há dados suficientes para criar uma sequência de {time_steps} dias para predição (API). Dados disponíveis: {ultimos_dias_scaled.shape[0]} dias.")
            return None # Retornar None se dados insuficientes

        X_input = np.reshape(ultimos_dias_scaled, (1, ultimos_dias_scaled.shape[0], 1))
        previsao_escalada = MODEL.predict(X_input) # Usando o MODEL global
        previsao_desnormalizada = SCALER.inverse_transform(previsao_escalada) # Desnormalizando com o SCALER global
        predicted_price = previsao_desnormalizada[0][0] # Extraindo o valor escalar da predição

        logging.info(f'Predição para {symbol} realizada com sucesso (API). Preço previsto: {predicted_price}') # Log com preço previsto
        return {'predicted_price': float(predicted_price)} # Retornando um dicionário JSON-serializável

    except Exception as e:
        logging.error(f'Erro ao realizar a predição para {symbol} (API): {e}')
        raise RuntimeError(f'Erro ao realizar a predição para API: {e}')


if __name__ == '__main__':
    configure_logging()

    # Exemplo de uso (para teste fora da API)
    try:
        load_scaler_for_api() # Carrega o scaler uma vez
        load_model_for_api() # Carrega o modelo uma vez

        symbol = 'MSFT'
        start_date = '2024-01-01'
        end_date = '2024-02-20'
        prediction_result = predict_price_for_api(symbol, start_date, end_date) # Chama a função de predição para API

        if prediction_result:
            print(f"Preço previsto para {symbol} ({start_date} - {end_date}): {prediction_result['predicted_price']:.2f}")
        else:
            print(f"Falha ao obter a predição para {symbol} ({start_date} - {end_date}).")

    except RuntimeError as e:
        print(f"Erro no exemplo de uso: {e}")