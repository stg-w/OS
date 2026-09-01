"""
Mortality Classifier Module
Trains, calibrates (Isotonic), and evaluates Random Forest on ICU patient vitals.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from src.config import FEATURES, TARGET, RANDOM_STATE, DEFAULT_DATASET_PATH


class MortalityClassifier:
    """Calibrated Random Forest model for ICU patient mortality probability estimation."""

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path
        self.df = None
        self.base_model = None
        self.calibrated_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_meta = None
        self.test_probabilities = None
        self.metrics = {}
        self.is_trained = False

    def load_and_train(self):
        """Loads ICU dataset, splits with stratification, and trains calibrated model."""
        if self.is_trained:
            return

        self.df = pd.read_csv(self.dataset_path)
        X = self.df[FEATURES]
        y = self.df[TARGET]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        self.test_meta = self.df.loc[self.X_test.index].reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)

        # 1. Base Random Forest
        self.base_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        )
        self.base_model.fit(self.X_train, self.y_train)

        # 2. Calibrated Isotonic Classifier for true clinical risk estimates
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model, method="isotonic", cv=5
        )
        self.calibrated_model.fit(self.X_train, self.y_train)

        # 3. Evaluate
        predictions = self.calibrated_model.predict(self.X_test)
        probs = self.calibrated_model.predict_proba(self.X_test)[:, 1]
        self.test_probabilities = probs

        self.metrics = {
            "accuracy": float(accuracy_score(self.y_test, predictions)),
            "roc_auc": float(roc_auc_score(self.y_test, probs)),
            "pr_auc": float(average_precision_score(self.y_test, probs)),
            "report": classification_report(self.y_test, predictions, output_dict=True),
            "confusion_matrix": confusion_matrix(self.y_test, predictions).tolist(),
            "feature_importances": dict(
                zip(FEATURES, [float(v) for v in self.base_model.feature_importances_])
            ),
        }
        self.is_trained = True

    def predict_mortality_risk(self, vitals_dict: dict) -> float:
        """Predict calibrated mortality risk probability for given patient vitals."""
        if not self.is_trained:
            self.load_and_train()
        row_df = pd.DataFrame([{f: float(vitals_dict.get(f, 0.0)) for f in FEATURES}])
        prob = float(self.calibrated_model.predict_proba(row_df)[:, 1][0])
        return prob
