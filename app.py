import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import HeartDiseasePredictor
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize API and load the model
app = FastAPI(title="Heart Disease Prediction API", version="1.0")
predictor = HeartDiseasePredictor(model_path="heart_disease_dataset/model_artifacts/best_model.pkl")

# This instantly hooks into all API routes and exposes a /metrics dashboard
Instrumentator().instrument(app).expose(app)

# Define the expected JSON payload schema
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "Heart Disease Risk Classifier"}

@app.post("/predict")
def predict_risk(data: PatientData):
    """
    Accepts raw patient data, validates it, and returns the risk prediction.
    """
    try:
        # data.model_dump() converts the Pydantic object back into a standard Python dictionary
        result = predictor.predict(data.model_dump())
        
        # Format the response to exactly match the assignment requirements
        return {
            "prediction": result["heart_disease_risk"],
            "confidence": result["model_estimated_probability"],
            "diagnosis": result["diagnosis"]
        }
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        # If the predictor's strict validation fails, return a 400 Bad Request to the client
        raise HTTPException(status_code=400, detail=str(e))