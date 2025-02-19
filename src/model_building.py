import logging
from src.logger import configure_logging

from src.data_handler import download_stock_data, preprocess_data, standardize_data
from src.lstm_model import create_model, save_model

from datetime import datetime



def main(symbol: str):
    """
    Executa o pipeline completo para criar e salvar um modelo de previsão para uma ação específica.

    Este pipeline inclui as etapas de:
    1. Download de dados históricos da ação.
    2. Padronização dos dados.
    3. Pré-processamento dos dados para treinamento do modelo.
    4. Criação e treinamento do modelo de machine learning.
    5. Salvamento do modelo treinado.

    Args:
        symbol (str): O símbolo da ação para a qual o modelo será criado (ex: "AAPL", "MSFT").

    Raises:
        RuntimeError: Se ocorrer qualquer erro durante a execução do pipeline,
                      uma exceção RuntimeError será levantada com uma mensagem detalhada do erro.
    """
    try:
        logging.info(f'Iniciando pipeline para criar modelo para a ação: {symbol}.')

        start_date = '2020-01-01'
        end_date = '2025-02-19'
        logging.info(f'Baixando dados de {symbol} de {start_date} até {end_date}.')
        data = download_stock_data(stock_symbol=symbol, start=start_date, end=end_date)

        logging.info(f'Padronizando os dados de {symbol}.')
        data = standardize_data(data)

        logging.info(f'Pré-processando os dados de {symbol} para treinamento.')
        X_train, y_train, X_test, y_test = preprocess_data(data=data)

        logging.info(f'Criando e treinando o modelo para {symbol}.')
        model, history = create_model(X_train, y_train, X_test, y_test)

        logging.info(f'Salvando o modelo.')
        save_model(model)

        logging.info(f'Modelo criado e salvo com sucesso!')

    except Exception as e:
        error_message = f'Erro ao executar o pipeline para a ação {symbol}: {e}'
        logging.error(error_message)
        raise RuntimeError(error_message) from e  # Relevanta a exceção original encadeada

if __name__ == '__main__':
    main('MSFT')