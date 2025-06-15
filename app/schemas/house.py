from pydantic import BaseModel

class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad: int
    guestroom: int
    basement: int
    hotwaterheating: int
    airconditioning: int

class PredictionResponse(BaseModel):
    price: float
