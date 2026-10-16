import json
import logging
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
import joblib
import sklearn

import mlflow
import mlflow.sklearn

# Configure logging for pipeline observability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class HeartDiseaseModelPipeline:
    """
    Handles feature engineering, training, hyperparameter tuning, 
    strict evaluation, and MLflow experiment tracking for the Heart Disease dataset.
    """
    def __init__(self, raw_data_path: str = "heart_disease_dataset/processed.cleveland.data"):
        # Log environment version to audit dependency compatibility
        logging.info(f"Initializing inference with scikit-learn version: {sklearn.__version__}")
        self.raw_data_path = Path(raw_data_path)
        self.columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        self.categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        self.numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        
        self.random_state = 42
        self.test_split_proportion = 0.2
        self.best_model = None
        self.best_model_name: str = ""
        self.best_run_id: str = ""
        self.results_summary: dict[str, Any] = {}
        
        # Configure MLflow tracking
        self.artifact_dir = Path("heart_disease_dataset/model_artifacts")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Heart_Disease_Prediction")

    def load_and_split_data(self):
        """Loads raw data, formally validates the target, and performs stratified train/test split."""
        try:
            logging.info("Loading raw data for model training...")
            data = pd.read_csv(self.raw_data_path, names=self.columns, na_values=["?"])
            
            data["target"] = pd.to_numeric(data["target"], errors="raise")
            if data["target"].isna().any():
                raise ValueError("Target contains missing values.")
            if not data["target"].isin([0, 1, 2, 3, 4]).all():
                raise ValueError("Unexpected target values found.")
            
            data["target"] = (data["target"] > 0).astype(int)
            
            X = data.drop(columns=["target"])
            y = data["target"]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_split_proportion, random_state=self.random_state, stratify=y
            )
            
            logging.info("Data split successful.")
            
        except Exception as e:
            logging.error(f"Failed to load or split data: {e}")
            raise

    def _build_preprocessor(self) -> ColumnTransformer:
        """Constructs the Scikit-Learn ColumnTransformer for feature engineering."""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

    def _plot_cv_metrics(self, name: str, metrics: dict, output_path: Path):
        """Generates a bar plot for CV metrics to log for all runs."""
        plt.figure(figsize=(8, 5))
        plt.bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
        plt.ylim(0, 1.0)
        plt.title(f"{name} - Mean CV Metrics")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def train_and_evaluate(self):
        """Builds pipelines, performs CV selection, and tracks experiments in MLflow."""
        try:
            preprocessor = self._build_preprocessor()

            classifiers = {
                "LogisticRegression": {
                    "model": LogisticRegression(max_iter=1000, random_state=self.random_state),
                    "params": {
                        "classifier__C": [0.01, 0.1, 1.0, 10.0],
                        "classifier__penalty": ["l2"]
                    }
                },
                "RandomForest": {
                    "model": RandomForestClassifier(random_state=self.random_state),
                    "params": {
                        "classifier__n_estimators": [50, 100, 200],
                        "classifier__max_depth": [None, 5, 10],
                        "classifier__min_samples_split": [2, 5]
                    }
                }
            }

            scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc'}
            best_recall = 0.0
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

            for name, config in classifiers.items():
                logging.info(f"Training and tracking {name} via MLflow...")
                
                with mlflow.start_run(run_name=name) as run:
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', config["model"])
                    ])
                    
                    grid_search = GridSearchCV(
                        pipeline, param_grid=config["params"], cv=cv, 
                        scoring=scoring, refit='recall', n_jobs=-1
                    )
                    grid_search.fit(self.X_train, self.y_train)
                    
                    best_idx = grid_search.best_index_
                    
                    # 1. Log Experiment Configuration & Model Hyperparameters
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_params({
                        "random_seed": self.random_state,
                        "cv_strategy": "StratifiedKFold",
                        "cv_splits": cv.n_splits,
                        "selection_metric": "recall",
                        "test_split_proportion": self.test_split_proportion
                    })

                    # 2. Extract and Log CV Means
                    mean_cv_metrics = {metric: grid_search.cv_results_[f'mean_test_{metric}'][best_idx] for metric in scoring.keys()}
                    mlflow.log_metrics({f"cv_mean_{k}": v for k, v in mean_cv_metrics.items()})
                    
                    # 3. Extract and Log CV Standard Deviations
                    std_cv_metrics = {metric: grid_search.cv_results_[f"std_test_{metric}"][best_idx] for metric in scoring.keys()}
                    mlflow.log_metrics({f"cv_std_{k}": float(v) for k, v in std_cv_metrics.items()})
                    
                    # 4. Log Artifacts: CV Plot
                    plot_path = self.artifact_dir / f"{name}_cv_metrics.png"
                    self._plot_cv_metrics(name, mean_cv_metrics, plot_path)
                    mlflow.log_artifact(str(plot_path))
                    
                    # 5. Log Model Artifact securely using skops with explicitly trusted types
                    best_pipeline = grid_search.best_estimator_
                    mlflow.sklearn.log_model(
                        best_pipeline, 
                        "model", 
                        serialization_format="skops",
                        skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
                    )

                    self.results_summary[name] = {
                        "mean_cv_metrics": mean_cv_metrics,
                        "std_cv_metrics": std_cv_metrics,
                        "best_params": grid_search.best_params_,
                        "mlflow_run_id": run.info.run_id
                    }

                    # Select best overall model strictly based on mean CV recall
                    if mean_cv_metrics["recall"] > best_recall:
                        best_recall = mean_cv_metrics["recall"]
                        self.best_model = best_pipeline
                        self.best_model_name = name
                        self.best_run_id = run.info.run_id

            logging.info(f"Selected model: {self.best_model_name} (Recall: {best_recall:.4f})")

            # Strict hold-out evaluation on the selected champion
            self._evaluate_champion()

        except Exception as e:
            logging.error(f"Model training pipeline failed: {e}")
            raise

    def _evaluate_champion(self):
        """Evaluates the champion on the test set and appends results to its MLflow run."""
        logging.info(f"Evaluating champion ({self.best_model_name}) on hold-out test set...")
        
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        hold_out_metrics = {
            "test_accuracy": accuracy_score(self.y_test, y_pred),
            "test_precision": precision_score(self.y_test, y_pred, zero_division=0),
            "test_recall": recall_score(self.y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(self.y_test, y_proba)
        }
        
        # Reopen the winning MLflow run to attach test metrics and plots
        with mlflow.start_run(run_id=self.best_run_id):
            mlflow.log_metrics(hold_out_metrics)
            
            # Confusion Matrix
            cm_display = ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred, cmap="Blues")
            cm_path = self.artifact_dir / "test_confusion_matrix.png"
            plt.title(f"{self.best_model_name} - Test Confusion Matrix")
            plt.savefig(cm_path)
            mlflow.log_artifact(str(cm_path))
            plt.close()
            
            # ROC Curve
            roc_display = RocCurveDisplay.from_predictions(self.y_test, y_proba)
            roc_path = self.artifact_dir / "test_roc_curve.png"
            plt.title(f"{self.best_model_name} - Test ROC Curve")
            plt.savefig(roc_path)
            mlflow.log_artifact(str(roc_path))
            plt.close()

        self.results_summary["experiment_metadata"] = {
            "random_seed": self.random_state,
            "cv_strategy": "StratifiedKFold",
            "cv_splits": 5,
            "test_split_proportion": self.test_split_proportion,
            "selection_metric": "recall",
            "selected_model": self.best_model_name,
            "final_hold_out_metrics": hold_out_metrics
        }
        
        results_path = self.artifact_dir / "model_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results_summary, f, indent=4)
        logging.info(f"Model evaluation complete. Run 'mlflow ui' to view tracking dashboard.")

    def export_model(self):
        """Serializes the complete best trained pipeline for deployment."""
        if not self.best_model:
            raise ValueError("No trained model available to export.")
        try:
            out_path = self.artifact_dir / "best_model.pkl"
            joblib.dump(self.best_model, out_path)
            logging.info(f"Model pipeline explicitly exported to: {out_path}")
        except Exception as e:
            logging.error(f"Failed to export model: {e}")
            raise

if __name__ == "__main__":
    local_file_location = "heart_disease_dataset/processed.cleveland.data"
    
    trainer = HeartDiseaseModelPipeline(raw_data_path=local_file_location)
    trainer.load_and_split_data()
    trainer.train_and_evaluate()
    trainer.export_model()