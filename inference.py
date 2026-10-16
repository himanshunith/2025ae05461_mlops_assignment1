import logging
import pandas as pd
import numpy as np
import joblib
import sklearn
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class HeartDiseasePredictor:
    """
    Loads the serialized model pipeline and serves predictions on raw patient data.
    """
    def __init__(self, model_path: str = "heart_disease_dataset/model_artifacts/best_model.pkl"):
        self.model_path = Path(model_path)
        
        # Log environment version to audit dependency compatibility
        logging.info(f"Initializing inference with scikit-learn version: {sklearn.__version__}")
        
        # Define expected schema and boundaries for strict input validation
        self.expected_features = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        self.expected_categories = {
            "sex": {0.0, 1.0},
            "cp": {1.0, 2.0, 3.0, 4.0},
            "fbs": {0.0, 1.0},
            "restecg": {0.0, 1.0, 2.0},
            "exang": {0.0, 1.0},
            "slope": {1.0, 2.0, 3.0},
            "ca": {0.0, 1.0, 2.0, 3.0},
            "thal": {3.0, 6.0, 7.0}
        }
        
        self.pipeline = self._load_model()

    def _load_model(self):
        """Deserializes the saved scikit-learn pipeline and validates its interface."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        try:
            logging.info(f"Loading packaged model pipeline from {self.model_path}...")
            model = joblib.load(self.model_path)
            
            # Validate the loaded artifact exposes the expected scikit-learn interface
            required_methods = ["predict", "predict_proba", "classes_"]
            if not all(hasattr(model, attr) for attr in required_methods):
                raise TypeError(f"Loaded artifact lacks required inference interface: {required_methods}")
                
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading error: {e}")

    def _validate_input(self, patient_data: dict):
        """Validates schema completeness, numeric types, finite boundaries, and categorical sets."""
        missing_features = set(self.expected_features) - set(patient_data.keys())
        extra_features = set(patient_data.keys()) - set(self.expected_features)
        
        if missing_features:
            raise ValueError(f"Input missing required features: {missing_features}")
        if extra_features:
            raise ValueError(f"Input contains unexpected features: {extra_features}")

        for feature, value in patient_data.items():
            # Type Check
            if not isinstance(value, (int, float)):
                raise TypeError(f"Feature '{feature}' must be numeric. Received: {type(value)}")
            
            # Finite Check (Rejects NaN and Inf)
            if not np.isfinite(value):
                raise ValueError(f"Feature '{feature}' must be a finite number. Received: {value}")
            
            # Categorical Boundaries Check
            if feature in self.expected_categories:
                if float(value) not in self.expected_categories[feature]:
                    raise ValueError(f"Feature '{feature}' value {value} is not in expected categories {self.expected_categories[feature]}")

    def predict(self, patient_data: dict) -> dict:
        """
        Validates raw patient data, applies the embedded preprocessing, 
        and returns the binary prediction and risk probability.
        """
        # 1. Validate Input (Input Error Handling)
        try:
            self._validate_input(patient_data)
        except (ValueError, TypeError) as validation_err:
            logging.error(f"Input validation failed: {validation_err}")
            raise

        # 2. Execute Inference (Runtime Error Handling)
        try:
            df = pd.DataFrame([patient_data])
            logging.info("Processing validated features through the embedded ColumnTransformer...")
            
            prediction = self.pipeline.predict(df)[0]
            
            # Dynamically locate the index for the positive class (1)
            class_1_idx = np.where(self.pipeline.classes_ == 1)[0][0]
            probability = self.pipeline.predict_proba(df)[0][class_1_idx]
            
            result = {
                "heart_disease_risk": int(prediction),
                "model_estimated_probability": round(float(probability), 4),
                "diagnosis": "Elevated predicted risk" if prediction == 1 else "Lower predicted risk"
            }
            
            logging.info(f"Prediction successful: {result['diagnosis']} (Estimated Probability: {result['model_estimated_probability']*100:.2f}%)")
            return result

        except Exception as runtime_err:
            logging.error(f"Inference failed during execution: {runtime_err}")
            raise RuntimeError(f"Model prediction error: {runtime_err}")

if __name__ == "__main__":
    # Example: A raw, unscaled patient record mimicking the UCI format
    sample_patient = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 3.0,
        "ca": 0.0,
        "thal": 6.0
    }

    predictor = HeartDiseasePredictor()
    prediction_result = predictor.predict(sample_patient)
    print("\n--- Final Output ---")
    print(prediction_result)