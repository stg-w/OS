"""
LivingMemoryOS v4 - CLI Benchmark Runner (MIMIC-IV Clinical Cohort)
Self-Evolving Clinical Memory Architecture with Prognostic Retention and
Patient-Criticality Inheritance for Healthcare Edge Devices.
Run with: python main_new.py
"""

import time
import pandas as pd
from src.config_new import (
    FEATURES,
    DEFAULT_CAPACITY,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_PROTECTED_THRESHOLDS,
    EMERGENCY_CMS_THRESHOLD,
    RESULTS_V4_PATH,
    CAPACITY_V4_PATH,
)
from src.models.mimic_classifier import MimicMortalityClassifier
from src.memory.mimic_simulator import MimicLivingMemorySimulator


def run_benchmark():
    print("=" * 60)
    print(" LivingMemoryOS v4 - MIMIC-IV Clinical Benchmark")
    print(" 550,818 Inpatient Admissions & ICU Stays Cohort Engine")
    print("=" * 60)

    # 1. Model Training & Evaluation
    print("\n[1/3] Loading MIMIC-IV dataset & initializing clinical classifier...")
    t0 = time.time()
    classifier = MimicMortalityClassifier()
    classifier.load_and_train()
    print(f"Classifier ready in {round(time.time() - t0, 2)}s.")

    metrics = classifier.metrics
    print("\n" + "-" * 50)
    print("MODEL EVALUATION METRICS (In-Hospital Mortality)")
    print("-" * 50)
    print(f"Accuracy : {metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"ROC-AUC  : {metrics.get('roc_auc', 0.0):.4f}")
    print(f"PR-AUC   : {metrics.get('pr_auc', 0.0):.4f}")
    print("Confusion Matrix:\n", metrics.get("confusion_matrix"))
    print("\nFeature Importances:")
    for feat, imp in sorted(metrics.get("feature_importances", {}).items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<18}: {imp:.4f}")

    # 2. Simulation
    print("\n[2/3] Executing LivingMemoryOS v4 vs. Baseline FIFO Stream Simulation...")
    sim = MimicLivingMemorySimulator(classifier)
    sim_res = sim.run_simulation(
        memory_size=DEFAULT_CAPACITY,
        stream_size=10000,
        cms_weights=DEFAULT_CMS_WEIGHTS,
        protected_thresholds=DEFAULT_PROTECTED_THRESHOLDS,
        emergency_threshold=EMERGENCY_CMS_THRESHOLD,
    )

    # 3. Print Results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Stream Size Evaluated       : {sim_res['stream_size']:,} clinical pages")
    print(f"Base Physical Memory Size   : {sim_res['memory_size']} slots")
    print(f"Self-Evolving Capacity      : {sim_res['adaptive_capacity']} slots")
    print(f"FIFO Baseline Average CMS   : {sim_res['fifo_average_cms']:.4f}")
    print(f"LivingMemoryOS Average CMS  : {sim_res['living_average_cms']:.4f}")
    print(f"Retention Improvement       : +{sim_res['improvement_pct']}%")
    print(f"Protected Non-Evictable     : {sim_res['protected_pages']} / {sim_res['memory_size']}")
    print(f"High Priority Pages         : {sim_res['high_priority_pages']}")
    print(f"Emergency Pages (CMS >= 0.8): {sim_res['emergency_pages']}")

    # 4. Save CSV outputs
    print("\n[3/3] Exporting in-memory pages & capacity evolution...")
    results_df = pd.DataFrame(sim_res["living_memory"])
    results_df.to_csv(RESULTS_V4_PATH, index=False)

    cap_df = pd.DataFrame([{
        "initial_capacity": sim_res["memory_size"],
        "adaptive_capacity": sim_res["adaptive_capacity"],
        "emergency_pages": sim_res["emergency_pages"],
        "protected_pages": sim_res["protected_pages"],
    }])
    cap_df.to_csv(CAPACITY_V4_PATH, index=False)

    print(f"Saved: {RESULTS_V4_PATH} ({len(results_df)} pages)")
    print(f"Saved: {CAPACITY_V4_PATH}")
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()

