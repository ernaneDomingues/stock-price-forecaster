from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/predict")
async def predict_price(symbol: str):
    if model is None:
        return JSONResponse(content={"error": "Modelo não carregado. Verifique os logs de inicialização."}, status_code=500)
    
    try:
        return JSONResponse(content={"symbol": symbol, "predicted_price": float(predicted_price)}, status_code=200)

    except Exception as e:
        print(f"Erro ao processar o símbolo {symbol}: {e}") # Log do erro
        return JSONResponse(content={"error": f"Erro ao processar o símbolo {symbol}: {str(e)}"}, status_code=500)