import joblib
import numpy as np
from app.schemas.house import HouseFeatures

model = joblib.load('app/models/model.pkl')

FEATURE_ORDER = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning'
]

def predict_price(features: HouseFeatures) -> float:
    data = np.array([[getattr(features, f) for f in FEATURE_ORDER]])
    price = model.predict(data)[0]
    return float(price)
