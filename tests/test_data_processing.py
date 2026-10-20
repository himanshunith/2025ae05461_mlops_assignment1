import pytest
import numpy as np
import tempfile
from pathlib import Path
from data_pipeline import HeartDiseaseDataPipeline

def test_data_acquisition_and_target_mapping():
    """Verifies the schema and ensures the target is strictly mapped to 0 and 1."""
    pipeline = HeartDiseaseDataPipeline()
    df = pipeline.acquire_data()
    
    assert df.shape[1] == 14, "Dataset should contain exactly 14 columns."
    assert "target" in df.columns, "Target column is missing."
    assert set(df["target"].unique()).issubset({0, 1}), "Target mapping failed: contains values other than 0 and 1."

def test_data_cleaning_imputation():
    """Verifies missing values are successfully imputed using a temporary test directory."""
    pipeline = HeartDiseaseDataPipeline()
    df = pipeline.acquire_data()
    
    # Artificially inject missing values for testing
    df.loc[0, "age"] = np.nan
    df.loc[0, "sex"] = np.nan
    pipeline.data = df
    
    # Use a temporary directory to isolate test artifacts
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_output_path = Path(tmpdirname) / "test_cleaned_eda.csv"
        cleaned_df = pipeline.clean_for_eda(str(test_output_path))
        
        assert test_output_path.exists(), "EDA artifact was not saved."
        assert not cleaned_df["age"].isna().any(), "Continuous imputation failed: NaN remains in 'age'."
        assert not cleaned_df["sex"].isna().any(), "Categorical imputation failed: NaN remains in 'sex'."