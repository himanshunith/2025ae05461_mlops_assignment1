import logging
from typing import Optional
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging for pipeline observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class HeartDiseaseDataPipeline:
    """
    Robust data pipeline for acquiring, cleaning, preprocessing, 
    and conducting EDA on the UCI Heart Disease dataset.
    """
    def __init__(self, data_path: Optional[str] = "heart_disease_dataset/processed.cleveland.data"):
        self.data_path = Path(data_path) if data_path else None
        self.remote_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        self.data: Optional[pd.DataFrame] = None
        self.columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        
        self.categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        self.numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

        # Expected unique values for categorical validation
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

    def acquire_data(self) -> pd.DataFrame:
        """
        Loads dataset from local filesystem if available. If not, downloads the exact 
        raw bytes from the UCI repository for strict reproducibility, then loads it.
        """
        try:
            # Download byte-for-byte raw artifact if it doesn't exist locally
            if self.data_path and not self.data_path.exists():
                logging.info(f"Local file not found at '{self.data_path}'. Downloading exact raw bytes from remote URL...")
                self.data_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(self.remote_url, self.data_path)
                logging.info(f"Downloaded and cached exact raw dataset locally to: {self.data_path}")
            elif self.data_path and self.data_path.exists():
                logging.info(f"Loading local dataset from: {self.data_path}")

            # Load the data, explicitly handling '?' as NaN to enforce correct dtypes
            read_kwargs = {"names": self.columns, "na_values": ["?"]}
            source = self.data_path if self.data_path and self.data_path.exists() else self.remote_url
            self.data = pd.read_csv(source, **read_kwargs)

            if self.data.empty:
                raise ValueError("Loaded dataset is empty.")

            # Reliability Check: verify expected shape and missing values
            logging.info(f"Data acquired. Shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns.")
            missing_counts = self.data.isnull().sum()
            logging.info(f"Initial missing value counts:\n{missing_counts[missing_counts > 0]}")

            # Target conversion: 0 = healthy, 1-4 = presence of heart disease
            self.data["target"] = self.data["target"].apply(lambda x: 1 if int(x) > 0 else 0)
            return self.data

        except Exception as err:
            logging.error(f"Unexpected failure during data acquisition: {err}")
            raise

    def clean_for_eda(self, output_clean_path: str = "heart_disease_dataset/cleaned_heart_data_eda.csv") -> pd.DataFrame:
        """
        Cleans data strictly for Exploratory Data Analysis (EDA).
        WARNING: Not to be used for model training to prevent data leakage.
        """
        if self.data is None:
            raise ValueError("No data found. Call acquire_data() prior to cleaning.")

        try:
            logging.info("Starting data cleaning for EDA purposes only...")
            logging.warning("This artifact is for EDA only. Model training pipelines will handle imputations internally.")
            
            # Hard Validation: Reject unexpected categorical values
            for col, expected_vals in self.expected_categories.items():
                observed_vals = set(self.data[col].dropna().unique())
                anomalies = observed_vals - expected_vals
                if anomalies:
                    error_msg = f"Validation Error: Unexpected values {anomalies} found in categorical column '{col}'. Halting pipeline."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logging.info(f"Categorical column '{col}' validated successfully.")

            # Impute continuous numerical columns with median using direct assignment
            for col in self.numerical_cols:
                if self.data[col].isnull().sum() > 0:
                    median_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_val)
                    logging.info(f"Imputed missing entries in continuous column '{col}' with median: {median_val}")

            # Impute categorical columns with most frequent value (mode) using direct assignment
            for col in self.categorical_cols:
                if self.data[col].isnull().sum() > 0:
                    mode_val = self.data[col].mode()[0]
                    self.data[col] = self.data[col].fillna(mode_val)
                    logging.info(f"Imputed missing entries in categorical column '{col}' with mode: {mode_val}")

            clean_path = Path(output_clean_path)
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(clean_path, index=False)
            logging.info(f"EDA-ready dataset saved successfully to: {clean_path}")
            
            return self.data

        except Exception as err:
            logging.error(f"Error during preprocessing pipeline: {err}")
            raise

    def generate_eda_reports(self, output_dir: str = "heart_disease_dataset/eda_reports") -> None:
        """
        Produces and exports class balance, feature distribution histograms,
        and correlation heatmap charts.
        """
        if self.data is None:
            raise ValueError("Cannot perform EDA without loaded data.")

        try:
            logging.info(f"Generating EDA visual reports in directory: {output_dir}")
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            sns.set_theme(style="whitegrid")

            # 1. Target Class Balance
            plt.figure(figsize=(6, 4))
            ax = sns.countplot(data=self.data, x="target", hue="target", palette=["#2ecc71", "#e74c3c"], legend=False)
            plt.title("Heart Disease Target Distribution (0: Healthy, 1: Risk)")
            plt.xlabel("Diagnostic Target")
            plt.ylabel("Patient Count")
            for p in ax.patches:
                ax.annotate(f"{int(p.get_height())}", (p.get_x() + 0.35, p.get_height() + 2))
            plt.tight_layout()
            plt.savefig(out_path / "class_balance.png", dpi=300)
            plt.close()

            # 2. Continuous Numerical Histograms
            self.data[self.numerical_cols].hist(bins=15, figsize=(12, 8), color="steelblue", edgecolor="black")
            plt.suptitle("Distribution of Key Clinical Continuous Features")
            plt.tight_layout()
            plt.savefig(out_path / "feature_histograms.png", dpi=300)
            plt.close()

            # 3. Correlation Matrix Heatmap
            logging.warning("Generating correlation heatmap. Note: Pearson correlation on nominal categorical codes is for exploratory view only.")
            plt.figure(figsize=(12, 10))
            corr = self.data.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
            plt.title("Feature Correlation Matrix (Includes Categorical Labels)")
            plt.tight_layout()
            plt.savefig(out_path / "correlation_heatmap.png", dpi=300)
            plt.close()

            logging.info("EDA visualizations successfully created and saved.")

        except Exception as err:
            logging.error(f"Failed to generate EDA artifacts: {err}")
            raise

if __name__ == "__main__":
    local_file_location = "heart_disease_dataset/processed.cleveland.data"
    
    pipeline = HeartDiseaseDataPipeline(data_path=local_file_location)
    pipeline.acquire_data()
    pipeline.clean_for_eda()
    pipeline.generate_eda_reports()