from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.model_predict import predict_price_for_api
from pydantic import BaseModel, validator
from datetime import datetime
import logging

router = APIRouter()

class PredictionRequest(BaseModel):
    symbol: str = Query(..., description="Símbolo da ação para prever (ex: AAPL)")
    start_date: str = Query(..., description="Data de início para buscar dados (YYYY-MM-DD)")
    end_date: str = Query(..., description="Data de fim para buscar dados (YYYY-MM-DD)")

    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Data deve estar no formato YYYY-MM-DD")


@router.get("/predict", response_model=dict)
async def predict_endpoint(request: PredictionRequest):
    """
    Endpoint para prever o preço de fechamento de uma ação.
    Retorna um JSON com o preço previsto.
    """
    try:
        prediction_result = predict_price_for_api(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )

        if prediction_result is None:
            raise HTTPException(status_code=400, detail="Não foi possível obter a predição. Verifique se há dados disponíveis para o período e símbolo especificados.")

        return prediction_result # Retorna o dicionário com a predição (FastAPI converte para JSON)

    except RuntimeError as e:
        logging.error(f"Erro interno ao processar predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor ao realizar a predição.")
    except Exception as e:
        logging.error(f"Erro inesperado ao processar predição: {e}")
        raise HTTPException(status_code=500, detail="Erro inesperado no servidor.")