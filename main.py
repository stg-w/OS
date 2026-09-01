"""
LivingMemoryOS - Command Line Interface (CLI) Entrypoint
Run with: python main.py
"""

import pandas as pd
from src.models.classifier import MortalityClassifier
from src.memory.simulator import LivingMemorySimulator


def main():
    print("=" * 60)
    print("  LIVING MEMORY OS (CAMR) - CLINICAL REPLACEMENT ENGINE")
    print("=" * 60)

    # 1. Initialize and train model
    print("\n[1/3] Training Calibrated Random Forest Classifier...")
    classifier = MortalityClassifier()
    classifier.load_and_train()

    metrics = classifier.metrics
    print("\n" + "=" * 60)
    print("  MODEL VALIDATION METRICS")
    print("=" * 60)
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC   : {metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(" ", row)

    # 2. Run memory replacement simulation
    print("\n[2/3] Executing Page Replacement Simulation...")
    simulator = LivingMemorySimulator(classifier)
    sim = simulator.run_simulation()

    print("\n" + "=" * 60)
    print("  PAGE REPLACEMENT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"FIFO Critical Pages Retained          : {sim['fifo_critical_pages']} / {sim['fifo_size']}")
    print(f"LivingMemoryOS Critical Pages Retained: {sim['living_critical_pages']} / {len(sim['living_memory'])}")
    print(f"Emergency Pages Rejected (Capacity)   : {sim['rejected_emergency']}")

    print("\nTier Pool Allocations:")
    for tier, data in sim["tier_summary"].items():
        print(f" - {tier:14s}: {data['occupied']}/{data['capacity']} pages ({data['utilization_pct']}%)")

    if sim["fifo_critical_pages"] > 0:
        print(f"\nRetention Improvement over FIFO: +{sim['improvement_pct']}%")

    # 3. Save results to CSV
    print("\n[3/3] Exporting Results...")
    results_df = pd.DataFrame(sim["living_memory"])
    results_df.to_csv("livingmemory_results_v2.csv", index=False)
    print("Saved active memory pages to: livingmemory_results_v2.csv")
    print("\n[SUCCESS] Run completed.")


if __name__ == "__main__":
    main()