"""
Mortality Risk Prediction and AI Risk Generation Module

This module trains and evaluates a Random Forest machine learning model for
predicting in-hospital mortality using clinically engineered patient risk
factors. The workflow includes dataset loading, feature selection, stratified
train-test splitting, model training, performance evaluation, and probability
estimation. For each patient, the model generates an AI Risk score representing
the predicted likelihood of mortality based on age-related risk, ICU stay
severity, transfer escalation patterns, and laboratory abnormalities. The
resulting AI Risk scores are stored for use in subsequent LivingMemoryOS
processes, including Patient-Criticality Inheritance calculation, Clinical
Memory Score (CMS) computation, prognostic retention policies, patient
prioritization, and adaptive memory management simulations.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ==========================================
# LOAD DATA
# ==========================================

master = pd.read_csv(
    r"C:\Users\subiw\OS\data\master.csv",
    low_memory=False
)

# ==========================================
# FEATURES
# ==========================================

features = [
    "age_risk",
    "icu_risk",
    "escalation_risk",
    "biomarker_risk"
]

X = master[features]

y = master["hospital_expire_flag"]

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# EVALUATION
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n======================")
print("MODEL PERFORMANCE")
print("======================")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================================
# AI RISK SCORE
# ==========================================

master["ai_risk"] = (
    model.predict_proba(X)[:, 1]
)

print("\nAI Risk Statistics")

print(
    "Min:",
    round(master["ai_risk"].min(),4)
)

print(
    "Mean:",
    round(master["ai_risk"].mean(),4)
)

print(
    "Max:",
    round(master["ai_risk"].max(),4)
)

# ==========================================
# SAVE
# ==========================================

master.to_csv(
    r"C:\Users\subiw\OS\data\model_output.csv",
    index=False
)

print(
    "\nSaved: model_output.csv"
)

print(
    "Rows:",
    len(master)
)