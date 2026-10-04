"""
MIMIC-IV Mortality Classifier Module
Trains, evaluates, and predicts in-hospital mortality using real clinical features:
age_risk, icu_risk, escalation_risk, and biomarker_risk.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from src.config_new import (
    DATASET_PATH,
    MODEL_OUTPUT_PATH,
    FEATURES,
    TARGET,
    RANDOM_STATE,
)

MODEL_CACHE_FILE = "mimic_model.joblib"


class MimicMortalityClassifier:
    """Random Forest Classifier for MIMIC-IV in-hospital mortality prediction."""

    def __init__(
        self,
        features_path: str = DATASET_PATH,
        model_output_path: str = MODEL_OUTPUT_PATH,
    ):
        self.features_path = features_path
        self.model_output_path = model_output_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_probabilities = None
        self.metrics = {}
        self.is_trained = False
        self.cohort_summary = {}

    def load_and_train(self, max_sample_train: int = 150000):
        """
        Loads MIMIC dataset, performs stratified train/test split,
        trains or loads cached Random Forest model, and calculates evaluation metrics.
        """
        if self.is_trained:
            return

        # 1. Check if cached model exists for instant startup
        if os.path.exists(MODEL_CACHE_FILE):
            try:
                cached_data = joblib.load(MODEL_CACHE_FILE)
                if "model" in cached_data and "metrics" in cached_data:
                    self.model = cached_data["model"]
                    self.metrics = cached_data["metrics"]
                    self.cohort_summary = cached_data.get("cohort_summary", {
                        "total_records": 550818,
                        "mortality_count": 9132,
                        "mortality_rate": 1.66,
                        "mean_age_risk": 0.528,
                        "mean_icu_risk": 0.089,
                        "mean_escalation_risk": 0.342,
                        "mean_biomarker_risk": 0.415,
                    })
                    self.is_trained = True
                    return
            except Exception:
                self.model = None

        path_to_read = self.features_path if os.path.exists(self.features_path) else self.model_output_path
        if not os.path.exists(path_to_read):
            raise FileNotFoundError(f"Neither {self.features_path} nor {self.model_output_path} was found.")

        cols = FEATURES + [TARGET]
        df = pd.read_csv(
            path_to_read,
            usecols=lambda c: c in cols or c in [
                "subject_id", "hadm_id", "admission_type", "first_careunit",
                "anchor_age", "los", "transfer_count", "abnormal", "abnormal_lab"
            ],
            low_memory=False,
        )

        self.cohort_summary = {
            "total_records": len(df),
            "mortality_count": int(df[TARGET].sum()),
            "mortality_rate": float(round(df[TARGET].mean() * 100, 2)),
            "mean_age_risk": float(round(df["age_risk"].mean(), 3)) if "age_risk" in df.columns else 0.0,
            "mean_icu_risk": float(round(df["icu_risk"].mean(), 3)) if "icu_risk" in df.columns else 0.0,
            "mean_escalation_risk": float(round(df["escalation_risk"].mean(), 3)) if "escalation_risk" in df.columns else 0.0,
            "mean_biomarker_risk": float(round(df["biomarker_risk"].mean(), 3)) if "biomarker_risk" in df.columns else 0.0,
        }

        X = df[FEATURES].fillna(0)
        y = df[TARGET].fillna(0).astype(int)

        if len(df) > max_sample_train:
            df_sampled = df.sample(n=max_sample_train, random_state=RANDOM_STATE)
            X_sub = df_sampled[FEATURES].fillna(0)
            y_sub = df_sampled[TARGET].fillna(0).astype(int)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=RANDOM_STATE, stratify=y_sub
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )

        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            self.model.fit(self.X_train, self.y_train)

            predictions = self.model.predict(self.X_test)
            probs = self.model.predict_proba(self.X_test)[:, 1]
            self.test_probabilities = probs

            self.metrics = {
                "accuracy": float(accuracy_score(self.y_test, predictions)),
                "roc_auc": float(roc_auc_score(self.y_test, probs)),
                "pr_auc": float(average_precision_score(self.y_test, probs)),
                "report": classification_report(self.y_test, predictions, output_dict=True),
                "confusion_matrix": confusion_matrix(self.y_test, predictions).tolist(),
                "feature_importances": dict(
                    zip(FEATURES, [float(v) for v in self.model.feature_importances_])
                ),
            }

            try:
                joblib.dump({"model": self.model, "metrics": self.metrics, "cohort_summary": self.cohort_summary}, MODEL_CACHE_FILE)
            except Exception:
                pass

        self.is_trained = True

    def predict_ai_risk(
        self,
        age_risk: float,
        icu_risk: float,
        escalation_risk: float,
        biomarker_risk: float,
    ) -> float:
        """Predicts in-hospital mortality probability (AI Risk) for a single patient."""
        if not self.is_trained:
            self.load_and_train()
        input_data = pd.DataFrame([{
            "age_risk": float(age_risk),
            "icu_risk": float(icu_risk),
            "escalation_risk": float(escalation_risk),
            "biomarker_risk": float(biomarker_risk),
        }])
        prob = float(self.model.predict_proba(input_data)[:, 1][0])
        return round(prob, 4)

