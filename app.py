from fastapi import FastAPI
from src.logger import configure_logging
from routes import routes  # Importa o roteador definido em routes/routes.py
from src.model_predict import load_model_for_api, load_scaler_for_api
import logging

app = FastAPI(title="Stock Price Forecaster API", description="API para prever preços de ações usando modelo LSTM")

# Inclui as rotas definidas em routes/routes.py
app.include_router(routes.router)

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização da aplicação FastAPI.
    Carrega o modelo LSTM e o scaler ao iniciar a API.
    """
    configure_logging() # Configura o logging ao iniciar a aplicação
    logging.info("Iniciando a API...")
    try:
        load_scaler_for_api() # Carrega o scaler
        load_model_for_api() # Carrega o modelo
        logging.info("Modelo e Scaler carregados com sucesso!")
    except RuntimeError as e:
        logging.error(f"Erro durante a inicialização da API: {e}")
        # Em um cenário de produção, você poderia querer tratar esse erro de forma mais robusta,
        # como encerrar a aplicação ou tentar reconectar.
        raise e # Relevanta a exceção para que o FastAPI possa lidar com ela na inicialização

    logging.info("API iniciada e pronta para receber requisições.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) # Pass "app:app" as import string