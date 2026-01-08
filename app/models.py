from pydantic import BaseModel
from typing import Optional
from pydantic import ConfigDict


# ============================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ============================================================

class CustomerFeatures(BaseModel):
    """Model for customer input features"""
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int


class PredictionResponse(BaseModel):
    """Model for prediction response"""
    churn_probability: float
    prediction: int
    risk_level: str

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool    