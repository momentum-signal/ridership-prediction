from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from model.inference.predict_nbeats import NBeatsPredictor

app = FastAPI(
    title="N-BEATS Ridership Prediction API",
    description="API for predicting Komuter ridership using N-BEATS with confidence estimates.",
    version="1.0.0"
)

# Load model on startup
predictor = NBeatsPredictor(
    model_path="saved_models/nbeats-best.ckpt",
    training_data_path="data/cleaned_data.csv"
)

# Request model
class PredictionRequest(BaseModel):
    origin: str
    destination: str
    datetime: datetime

# Response model
class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    units: str = "riders"

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        prediction, error_pct = predictor.predict_with_confidence(
            dt=request.datetime,
            origin=request.origin,
            destination=request.destination
        )
        return {
            "prediction": prediction,
            "confidence": 1 - error_pct / 100.0,
            "units": "riders"
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# Endpoint to list valid stations
@app.get("/stations")
async def list_stations():
    return {
        "stations": list(predictor.station_encoder.keys())
    }

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
