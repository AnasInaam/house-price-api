# Entry point for FastAPI app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.house import HouseFeatures, PredictionResponse
from app.services.predict import predict_price
from fastapi import HTTPException

app = FastAPI()

# CORS setup (to be configured later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers here (to be added)

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        price = predict_price(features)
        return PredictionResponse(price=price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
