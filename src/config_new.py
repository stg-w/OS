"""
LivingMemoryOS v4 Configuration and System Parameters Module

This module centralizes all global configuration settings, dataset locations,
model inputs, scoring weights, protection thresholds, and adaptive memory
management parameters used throughout LivingMemoryOS v4. It provides a single
source of truth for file paths, machine learning feature definitions, Clinical
Memory Score (CMS) weight distributions, prognostic retention rules, and
self-evolving memory capacity controls. By isolating these constants from the
core implementation, the system becomes easier to maintain, tune, reproduce,
and deploy across different environments while ensuring consistent behavior
across all clinical scoring, prediction, simulation, and memory management
components.
"""

import os

RANDOM_STATE = 42

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def _resolve_data_path(filename: str) -> str:
    candidates = [
        os.path.join(_DATA_DIR, filename),
        os.path.join("data", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

# Dataset Paths
DATASET_PATH = "features.csv"
MODEL_OUTPUT_PATH = "model_output.csv"
RESULTS_V4_PATH = "livingmemory_results_v4.csv"
CAPACITY_V4_PATH = "capacity_evolution_v4.csv"
DATASET_PATH = _resolve_data_path("features.csv")
MODEL_OUTPUT_PATH = _resolve_data_path("model_output.csv")
RESULTS_V4_PATH = _resolve_data_path("livingmemory_results_new.csv") if os.path.exists(_resolve_data_path("livingmemory_results_new.csv")) else _resolve_data_path("livingmemory_results_v4.csv")
CAPACITY_V4_PATH = _resolve_data_path("capacity_evolution_new.csv") if os.path.exists(_resolve_data_path("capacity_evolution_new.csv")) else _resolve_data_path("capacity_evolution_v4.csv")

# Model Features & Target
FEATURES = [
    "age_risk",
    "icu_risk",
    "escalation_risk",
    "biomarker_risk",
]

TARGET = "hospital_expire_flag"

# Default Clinical Memory Score (CMS) Weights
DEFAULT_CMS_WEIGHTS = {
    "ai_risk": 0.30,
    "biomarker_risk": 0.20,
    "escalation_risk": 0.20,
    "icu_risk": 0.15,
    "age_risk": 0.10,
    "inheritance": 0.05,
}

# Prognostic Retention Thresholds
DEFAULT_PROTECTED_THRESHOLDS = {
    "ai_risk_cutoff": 0.75,
    "biomarker_risk_cutoff": 0.80,
}

# Edge Memory Capacity & Self-Evolving Rules
DEFAULT_CAPACITY = 100
ADAPTIVE_EXPANSION_STEP = 20
EMERGENCY_CMS_THRESHOLD = 0.80
ADAPTIVE_TRIGGER_COUNT = 30

