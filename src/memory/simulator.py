"""
LivingMemoryOS Simulation Engine
================================

Purpose
-------
Implements the core LivingMemoryOS memory-management simulation framework
for intelligent retention of clinical telemetry data.

Overview
--------
This module evaluates incoming patient telemetry streams, predicts
mortality risk using a machine-learning classifier, computes Clinical
Memory Scores (CMS), and simulates memory allocation under both
traditional FIFO and LivingMemoryOS replacement policies.

Unlike conventional memory managers that rely solely on arrival order,
LivingMemoryOS prioritizes memory retention according to clinical
importance, patient severity, and predicted risk.

Core Components
---------------
1. Mortality Risk Prediction
   - Uses a trained machine-learning model to estimate patient mortality.
   - Produces a calibrated criticality score for each patient.

2. Clinical Memory Scoring (CMS)
   - Combines mortality risk, physiological severity,
     biomarker abnormalities, and care-level importance.
   - Generates a memory-retention priority score.

3. Tier-Based Memory Architecture
   - EMERGENCY Tier
       Non-evictable storage reserved for critical patients.
   - HIGH_PRIORITY Tier
       Retains high-risk telemetry records.
   - NORMAL Tier
       Stores routine clinical monitoring data.

4. Tiered Min-Heap Replacement
   - Each tier maintains its own capacity constraints.
   - Lower-priority records are preferentially removed when space is needed.
   - Emergency records receive protection from eviction.

5. FIFO Benchmark Comparison
   - Executes a traditional FIFO memory simulation in parallel.
   - Measures the retention advantage provided by LivingMemoryOS.

Simulation Workflow
-------------------
1. Load test cohort telemetry data.
2. Predict mortality probabilities.
3. Calculate Clinical Memory Scores (CMS).
4. Assign care levels and retention tiers.
5. Simulate FIFO memory replacement.
6. Simulate LivingMemoryOS tiered memory allocation.
7. Compare retention effectiveness.
8. Generate performance metrics and dashboard outputs.

Key Metrics
-----------
The simulator reports:

- Critical patient retention
- Tier utilization statistics
- Emergency admission failures
- Memory occupancy
- FIFO comparison metrics
- Retention improvement percentage
- Clinical memory composition

Live Triage Support
-------------------
The simulator also supports real-time patient evaluation through
the triage_single_patient() interface, allowing incoming telemetry
streams to be immediately scored, classified, and assigned to the
appropriate memory tier.

Research Objective
------------------
Demonstrates that clinically-aware memory allocation can preserve
high-value patient information more effectively than traditional
operating-system replacement policies, improving retention of
critical telemetry under constrained memory budgets.

"""

from src.config import (
    DEFAULT_TIER_CAPACITY,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_CARE_LEVEL_THRESHOLDS,
    DEFAULT_CRITICAL_THRESHOLD,
)
from src.models.classifier import MortalityClassifier
from src.models.scoring import ClinicalScorer
from src.memory.tier_pool import TierPool
from src.memory.fifo_pool import FifoMemoryPool


class LivingMemorySimulator:
    """End-to-end memory replacement and patient telemetry simulation engine."""

    def __init__(self, classifier: MortalityClassifier = None):
        self.classifier = classifier or MortalityClassifier()
        if not self.classifier.is_trained:
            self.classifier.load_and_train()

    def build_page(
        self,
        idx: int,
        patient_id: str,
        vitals: dict,
        criticality: float,
        scorer: ClinicalScorer,
    ) -> dict:
        """Constructs a memory page record with metadata and calculated clinical scores."""
        care_level = scorer.assign_care_level(criticality)
        cms, severity, biomarker_risk = scorer.compute_cms(criticality, vitals, care_level)
        tier, reason = scorer.determine_tier_and_reasons(criticality, vitals, care_level)

        return {
            "page_id": idx,
            "patient_id": str(patient_id),
            "criticality": round(float(criticality), 4),
            "care_level": care_level,
            "severity": round(severity, 4),
            "biomarker_risk": round(biomarker_risk, 4),
            "sepsis": int(vitals.get("sepsis_flag", 0)),
            "cms": round(cms, 4),
            "tier": tier,
            "reason": reason,
            "vitals": vitals,
        }

    def run_simulation(
        self,
        tier_capacities: dict = None,
        cms_weights: dict = None,
        care_thresholds: dict = None,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    ) -> dict:
        """Executes full telemetry stream simulation across test cohort."""
        capacities = tier_capacities or DEFAULT_TIER_CAPACITY
        scorer = ClinicalScorer(
            care_thresholds=care_thresholds,
            cms_weights=cms_weights,
            critical_threshold=critical_threshold,
        )

        # 1. Evaluate incoming stream pages
        memory_pages = []
        for i in range(len(self.classifier.X_test)):
            row = self.classifier.X_test.iloc[i].to_dict()
            meta = self.classifier.test_meta.iloc[i]
            crit = float(self.classifier.test_probabilities[i])
            page = self.build_page(
                idx=i,
                patient_id=meta["patient_id"],
                vitals=row,
                criticality=crit,
                scorer=scorer,
            )
            memory_pages.append(page)

        # 2. FIFO baseline simulation
        fifo_size = sum(capacities.values())
        fifo_pool = FifoMemoryPool(capacity=fifo_size)
        for page in memory_pages:
            fifo_pool.admit(page)

        # 3. LivingMemoryOS Tiered Min-Heap simulation
        tier_pools = {
            "EMERGENCY": TierPool(capacities["EMERGENCY"], evictable=False),
            "HIGH_PRIORITY": TierPool(capacities["HIGH_PRIORITY"]),
            "NORMAL": TierPool(capacities["NORMAL"]),
        }

        rejected_emergency = 0
        for page in memory_pages:
            admitted = tier_pools[page["tier"]].admit(page)
            if not admitted and page["tier"] == "EMERGENCY":
                rejected_emergency += 1

        living_memory_pages = (
            tier_pools["EMERGENCY"].pages()
            + tier_pools["HIGH_PRIORITY"].pages()
            + tier_pools["NORMAL"].pages()
        )
        fifo_pages = fifo_pool.pages()

        # 4. Metrics & Comparison
        fifo_critical = sum(p["criticality"] >= critical_threshold for p in fifo_pages)
        living_critical = sum(p["criticality"] >= critical_threshold for p in living_memory_pages)

        improvement_pct = 0.0
        if fifo_critical > 0:
            improvement_pct = round(((living_critical - fifo_critical) / fifo_critical) * 100.0, 2)

        tier_summary = {
            name: {
                "occupied": pool.size(),
                "capacity": pool.capacity,
                "utilization_pct": round((pool.size() / pool.capacity) * 100.0, 1)
                if pool.capacity > 0
                else 0.0,
            }
            for name, pool in tier_pools.items()
        }

        return {
            "total_test_stream_size": len(memory_pages),
            "fifo_size": fifo_size,
            "fifo_critical_pages": fifo_critical,
            "fifo_critical_fraction": round(fifo_critical / max(1, len(fifo_pages)), 4),
            "living_critical_pages": living_critical,
            "living_critical_fraction": round(living_critical / max(1, len(living_memory_pages)), 4),
            "improvement_pct": improvement_pct,
            "rejected_emergency": rejected_emergency,
            "tier_summary": tier_summary,
            "living_memory": living_memory_pages,
            "fifo_memory": fifo_pages,
            "metrics": self.classifier.metrics,
        }

    def triage_single_patient(
        self,
        vitals: dict,
        patient_id: str = "LIVE_PT_01",
        cms_weights: dict = None,
        care_thresholds: dict = None,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    ) -> dict:
        """Evaluates a live incoming patient for instant mortality risk & tier assignment."""
        scorer = ClinicalScorer(
            care_thresholds=care_thresholds,
            cms_weights=cms_weights,
            critical_threshold=critical_threshold,
        )
        crit = self.classifier.predict_mortality_risk(vitals)
        return self.build_page(
            idx=999999,
            patient_id=patient_id,
            vitals=vitals,
            criticality=crit,
            scorer=scorer,
        )