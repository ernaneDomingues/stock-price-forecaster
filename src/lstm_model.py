from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import logging
from src.logger import configure_logging


def create_model(X_train, y_train, X_test, y_test, units=50, batch_size=32, epochs=100):
    """
    Cria, compila e treina um modelo LSTM para previsão de séries temporais.

    Args:
        X_train (numpy.ndarray): Dados de entrada para treinamento.
        y_train (numpy.ndarray): Rótulos de treinamento.
        X_test (numpy.ndarray): Dados de entrada para validação.
        y_test (numpy.ndarray): Rótulos de validação.
        units (int, optional): Número de unidades LSTM em cada camada. Padrão é 50.
        batch_size (int, optional): Tamanho do batch para treinamento. Padrão é 32.
        epochs (int, optional): Número de épocas para o treinamento. Padrão é 100.

    Returns:
        model: O modelo treinado.
        history: O histórico do treinamento.
    """
    
    def rmse(y_true, y_pred):
        """
        Função para calcular o erro quadrático médio (RMSE).
        """
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
    try:
        logging.info('Criando o modelo LSTM.')
        
        # EarlyStopping para interromper o treinamento se não houver melhoria
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Criando o modelo sequencial
        model = Sequential()

        # Primeira camada LSTM
        model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))

        # Segunda camada LSTM
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(0.2))

        # Camada densa de saída
        model.add(Dense(units=1))

        # Compilação do modelo com otimizador Adam e loss de MSE
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse', rmse])

        # Exibe a arquitetura do modelo
        model.summary()

        # Treinando o modelo
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=1,
                            callbacks=[early_stopping])

        logging.info('Modelo treinado com sucesso.')

        return model, history
    except Exception as e:
        logging.error(f'Erro ao treinar o modelo: {e}')
        raise RuntimeError(f'Erro ao treinar o modelo: {e}')

def save_model(model, symbol):
    """_summary_

    Args:
        model (_type_): _description_
        symbol (_type_): _description_

    Raises:
        RuntimeError: _description_
    """
    try:
        logging.info('Salvando o modelo.')
        model.save(fr'stock-price-forecaster\models\{symbol}_lstm_model.keras')
        logging.info('Modelo Salvo!')
    except Exception as e:
        logging.error(f'Erro ao salvar modelo: {e}')
        raise RuntimeError(f'Erro ao salvar modelo: {e}')