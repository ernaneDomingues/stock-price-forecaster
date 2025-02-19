import logging
from src.logger import configure_logging

from src.data_handler import download_stock_data, preprocess_data, standardize_data
from src.lstm_model import create_model, save_model

def main(symbol):
    """_summary_

    Args:
        symbol (_type_): _description_

    Raises:
        RuntimeError: _description_
    """
    try:
        logging.info('Pipeline para criar o modelo.')
        data = download_stock_data(stock_symbol=symbol)
        data = standardize_data(data)
        X_train, y_train, X_test, y_test = preprocess_data(data=data)
        model, history = create_model(X_train, y_train, X_test, y_test)
        save_model(model, symbol)
        logging.info('Modelo criado com sucesso!')
    except Exception as e:
        logging.error(f'Erro ao executar a pipeline: {e}')
        raise RuntimeError(f'Erro ao executar a pipeline: {e}')
    
if __name__ == '__main__':
    main('MSFT')