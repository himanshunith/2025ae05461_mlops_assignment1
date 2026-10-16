import pytest
import numpy as np
import json
from pathlib import Path
from inference import HeartDiseasePredictor

@pytest.fixture
def sample_valid_patient():
    return {
        "age": 63.0, "sex": 1.0, "cp": 1.0, "trestbps": 145.0, 
        "chol": 233.0, "fbs": 1.0, "restecg": 2.0, "thalach": 150.0, 
        "exang": 0.0, "oldpeak": 2.3, "slope": 3.0, "ca": 0.0, "thal": 6.0
    }

def test_model_evaluation_metrics():
    """Verifies the saved model results contain valid evaluation metrics between 0.0 and 1.0."""
    results_path = Path("heart_disease_dataset/model_artifacts/model_results.json")
    assert results_path.exists(), "Model results JSON not found. Did the training pipeline run?"
    
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    metrics = results["experiment_metadata"]["final_hold_out_metrics"]
    for metric_name in ["test_accuracy", "test_precision", "test_recall", "test_roc_auc"]:
        assert 0.0 <= metrics[metric_name] <= 1.0, f"Metric {metric_name} out of bounds: {metrics[metric_name]}"

def test_inference_missing_artifact():
    """Verifies the predictor gracefully rejects initialization when the model artifact is missing."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        HeartDiseasePredictor(model_path="invalid_path/non_existent_model.pkl")

def test_inference_rejects_nan_and_inf(sample_valid_patient):
    """Verifies the predictor explicitly rejects NaN and Infinity values during inference."""
    predictor = HeartDiseasePredictor()
    
    nan_patient = sample_valid_patient.copy()
    nan_patient["trestbps"] = np.nan
    with pytest.raises(ValueError, match="must be a finite number"):
        predictor.predict(nan_patient)
        
    inf_patient = sample_valid_patient.copy()
    inf_patient["chol"] = np.inf
    with pytest.raises(ValueError, match="must be a finite number"):
        predictor.predict(inf_patient)