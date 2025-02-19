from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from tensorflow.keras.utils import get_custom_objects, register_keras_serializable
import logging
from src.logger import configure_logging
import os # Importando os para usar caminhos relativos

@register_keras_serializable()
def rmse(y_true, y_pred):
    """
    Função para calcular o erro quadrático médio (RMSE).
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def create_model(X_train, y_train, X_test, y_test, units=50, batch_size=32, epochs=100, model_dir='models'): # Adicionando model_dir
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
        model_dir (str, optional): Diretório para salvar os modelos. Padrão é 'models'. # Novo argumento

    Returns:
        model: O modelo treinado.
        history: O histórico do treinamento.
    """



    try:
        logging.info('Criando o modelo LSTM.')

        # EarlyStopping para interromper o treinamento se não houver melhoria
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # ModelCheckpoint para salvar o melhor modelo durante o treinamento
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_lstm_model.keras'), # Caminho para salvar o melhor modelo
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False, # Salvar o modelo completo, não apenas os pesos
            verbose=1
        )
        callbacks = [early_stopping, model_checkpoint] # Usando ambos callbacks

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
                            callbacks=callbacks) # Usando a lista de callbacks

        logging.info('Modelo treinado com sucesso.')

        return model, history
    except Exception as e:
        logging.error(f'Erro ao treinar o modelo: {e}')
        raise RuntimeError(f'Erro ao treinar o modelo: {e}')

def save_model(model, model_dir='models'): # Adicionando model_dir
    """
    Salva o modelo treinado no caminho especificado, utilizando o símbolo no nome do arquivo.

    Args:
        modelo (keras.Model): O modelo treinado (por exemplo, um modelo LSTM) a ser salvo.
        model_dir (str, optional): Diretório para salvar os modelos. Padrão é 'models'. # Novo argumento

    Raises:
        RuntimeError: Se ocorrer um erro ao salvar o modelo.
    """
    try:
        logging.info(f'Salvando o modelo em {model_dir}') # Log com diretório e símbolo
        model_path = os.path.join(model_dir, f'lstm_model.keras') # Caminho relativo usando os.path.join e f-string correta
        model.save(model_path) # Salvando no caminho construído
        logging.info('Modelo Salvo!')
    except Exception as e:
        logging.error(f'Erro ao salvar modelo: {e}')
        raise RuntimeError(f'Erro ao salvar modelo: {e}')