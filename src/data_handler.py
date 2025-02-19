import time
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
from src.logger import configure_logging
import joblib
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import os
from datetime import datetime

def save_scaler(scaler_model, path):
    """Salva o modelo do scaler no caminho especificado.

    Args:
        scaler_model (object): O modelo do escalonador (por exemplo, StandardScaler, MinMaxScaler) a ser guardado.
        path (str): O caminho do ficheiro onde o modelo do scaler deve ser guardado.

    Raises:
        RuntimeError: Se ocorrer um erro ao guardar o modelo do scaler.
    """
    try:
        logging.info(f'Salvando o modelo scaler em: {path}')
        joblib.dump(scaler_model, path)
    except Exception as e:
        logging.error(f'Erro ao salvar o modelo scaler em {path}: {e}')
        raise RuntimeError(f'Erro ao salvar o modelo scaler: {e}')

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
        scaler_path = 'models/Scaler_model.pkl' # Caminho relativo
        save_scaler(scaler, path=scaler_path)
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

    
    
def download_stock_data(stock_symbol, start_date='2020-01-01', end_date=None, retry_delay=60, max_retries=3):
    """
    Baixa os dados históricos da ação, tentando yfinance primeiro e Alpha Vantage como fallback.

    Args:
        stock_symbol (str): O símbolo da ação (ex: "AAPL").
        start_date (str): A data inicial para o download dos dados. Formato: 'YYYY-MM-DD'.
        end_date (str, opcional): A data final para o download dos dados. Formato: 'YYYY-MM-DD'.
                                    Se None, usa a data de hoje.
        retry_delay (int): Tempo em segundos para esperar antes de retentar o download em caso de rate limit.
        max_retries (int): Número máximo de tentativas para baixar os dados com yfinance antes de usar Alpha Vantage.

    Returns:
        pandas.DataFrame: DataFrame com os dados históricos da ação, ou None em caso de falha após várias tentativas.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    if start_date_dt >= end_date_dt:
        logging.warning(f"Data de início ({start_date}) é igual ou posterior à data de fim ({end_date}). Retornando DataFrame vazio.")
        return pd.DataFrame()  # Retorna DataFrame vazio em vez de erro

    # Tentar com yfinance primeiro
    for retry in range(max_retries):
        try:
            logging.info(f'Tentativa {retry + 1}/{max_retries}: Iniciando download de {stock_symbol} com Yahoo Finance.')
            data = yf.download(stock_symbol, start=start_date, end=end_date, multi_level_index=False, progress=False)

            if not data.empty:
                data = data[['Close']]
                data.index = pd.to_datetime(data.index)
                logging.info(f'Dados de {stock_symbol} baixados com sucesso de Yahoo Finance.')
                return data
            else:
                logging.warning(f"Yahoo Finance retornou dados vazios para {stock_symbol} no período {start_date} - {end_date}.")
                break  # Sai do loop de retentativas e tenta Alpha Vantage

        except yf.YFError as yf_error: # Captura erros gerais de yfinance também
            if "Rate Limit Exceeded" in str(yf_error) or isinstance(yf_error, yf.YFRateLimitError): # Verifica explicitamente a mensagem ou o tipo de erro
                logging.warning(f"Yahoo Finance: Limite de requisições atingido para {stock_symbol} (Tentativa {retry + 1}/{max_retries}). Aguardando {retry_delay} segundos...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Erro com Yahoo Finance para {stock_symbol} (Tentativa {retry + 1}/{max_retries}): {yf_error}")
                break # Sai do loop de retentativas e tenta Alpha Vantage
        except Exception as e:
            logging.error(f"Erro inesperado ao usar Yahoo Finance para {stock_symbol} (Tentativa {retry + 1}/{max_retries}): {e}")
            break # Sai do loop de retentativas e tenta Alpha Vantage

    logging.info(f"Tentando baixar dados de {stock_symbol} com Alpha Vantage como fallback.")
    try:
        api_key = os.getenv('ALPHA_KEY')
        if not api_key:
            raise ValueError("Variável de ambiente ALPHA_KEY não configurada.")

        ts = TimeSeries(key=api_key, output_format='pandas')
        df, meta_data = ts.get_daily(symbol=stock_symbol, outputsize="full")

        if df.empty:
            logging.warning(f"Alpha Vantage retornou dados vazios para {stock_symbol}.")
            return pd.DataFrame() # Retorna DataFrame vazio se Alpha Vantage não retornar dados

        df = df[['4. close']].rename(columns={'4. close': 'Close'})
        df.index = pd.to_datetime(df.index)

        # Ajuste para data de início e fim (mais robusto)
        df_filtered = df[(df.index >= start_date_dt) & (df.index <= end_date_dt)]

        if df_filtered.empty:
            logging.warning(f"Alpha Vantage não retornou dados para {stock_symbol} no período {start_date} - {end_date}.")
            return pd.DataFrame() # Retorna DataFrame vazio se não houver dados no período desejado

        df_filtered = df_filtered.sort_index(ascending=True)
        logging.info(f'Dados de {stock_symbol} baixados com sucesso de Alpha Vantage.')
        return df_filtered

    except ValueError as ve:
        logging.error(f"Erro de configuração com Alpha Vantage para {stock_symbol}: {ve}")
        raise ValueError(f"Erro de configuração com Alpha Vantage: {ve}") # Re-levanta ValueError para indicar problema de config
    except Exception as e:
        logging.error(f"Erro ao processar download com Alpha Vantage para {stock_symbol}: {e}")
        raise RuntimeError(f"Falha ao baixar dados para {stock_symbol} com Yahoo Finance e Alpha Vantage: {e}") # Re-levanta RuntimeError para falha geral
