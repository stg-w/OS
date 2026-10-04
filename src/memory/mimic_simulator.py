"""
MIMIC-IV LivingMemory Simulator Module
Implements memory replacement simulation with prognostic retention,
victim selection (min-CMS of un-protected pages), adaptive capacity expansion,
and comparison against baseline FIFO.
"""

import os
import pandas as pd
import numpy as np
from src.config_new import (
    MODEL_OUTPUT_PATH,
    DATASET_PATH,
    DEFAULT_CAPACITY,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_PROTECTED_THRESHOLDS,
    EMERGENCY_CMS_THRESHOLD,
    ADAPTIVE_EXPANSION_STEP,
    ADAPTIVE_TRIGGER_COUNT,
)
from src.models.mimic_classifier import MimicMortalityClassifier
from src.models.mimic_scoring import MimicClinicalScorer


class MimicLivingMemorySimulator:
    """Simulator engine for LivingMemoryOS v4 on the MIMIC-IV Clinical Cohort."""

    def __init__(
        self,
        classifier: MimicMortalityClassifier = None,
        scorer: MimicClinicalScorer = None,
    ):
        self.classifier = classifier or MimicMortalityClassifier()
        if not self.classifier.is_trained:
            self.classifier.load_and_train()
        self.scorer = scorer or MimicClinicalScorer()

    def load_stream_pages(self, max_pages: int = 10000) -> list:
        path_to_read = MODEL_OUTPUT_PATH if os.path.exists(MODEL_OUTPUT_PATH) else DATASET_PATH
        if not os.path.exists(path_to_read):
            raise FileNotFoundError(f"Neither {MODEL_OUTPUT_PATH} nor {DATASET_PATH} exists.")

        df = pd.read_csv(
            path_to_read,
            nrows=max_pages,
            low_memory=False,
        )

        if "ai_risk" not in df.columns:
            X = df[["age_risk", "icu_risk", "escalation_risk", "biomarker_risk"]].fillna(0)
            df["ai_risk"] = self.classifier.model.predict_proba(X)[:, 1]

        df["age_risk"] = df["age_risk"].fillna(0).astype(float)
        df["icu_risk"] = df["icu_risk"].fillna(0).astype(float)
        df["escalation_risk"] = df["escalation_risk"].fillna(0).astype(float)
        df["biomarker_risk"] = df["biomarker_risk"].fillna(0).astype(float)
        df["ai_risk"] = df["ai_risk"].fillna(0).astype(float)

        return df.to_dict(orient="records")

    def run_simulation(
        self,
        memory_size: int = DEFAULT_CAPACITY,
        stream_size: int = 5000,
        cms_weights: dict = None,
        protected_thresholds: dict = None,
        emergency_threshold: float = EMERGENCY_CMS_THRESHOLD,
        adaptive_expansion_step: int = ADAPTIVE_EXPANSION_STEP,
        adaptive_trigger_count: int = ADAPTIVE_TRIGGER_COUNT,
    ) -> dict:
        scorer = MimicClinicalScorer(
            cms_weights=cms_weights,
            protected_thresholds=protected_thresholds,
            emergency_threshold=emergency_threshold,
        )

        raw_pages = self.load_stream_pages(max_pages=stream_size)

        scored_pages = []
        for i, p in enumerate(raw_pages):
            eval_res = scorer.evaluate_patient(
                ai_risk=float(p.get("ai_risk", 0.0)),
                age_risk=float(p.get("age_risk", 0.0)),
                icu_risk=float(p.get("icu_risk", 0.0)),
                escalation_risk=float(p.get("escalation_risk", 0.0)),
                biomarker_risk=float(p.get("biomarker_risk", 0.0)),
            )

            record = dict(p)
            record["page_id"] = i
            record["inheritance"] = eval_res["inheritance"]
            record["cms"] = eval_res["cms"]
            record["protected"] = eval_res["protected"]
            record["tier"] = eval_res["tier"]
            record["reason"] = eval_res["reason"]
            scored_pages.append(record)

        # Baseline FIFO
        fifo = []
        for page in scored_pages:
            if len(fifo) >= memory_size:
                fifo.pop(0)
            fifo.append(page)

        # LivingMemoryOS v4 with prognostic retention
        living = []
        rejected_count = 0
        eviction_count = 0

        for page in scored_pages:
            if len(living) < memory_size:
                living.append(page)
            else:
                candidates = [p for p in living if not p["protected"]]
                if len(candidates) == 0:
                    rejected_count += 1
                    continue

                victim = min(candidates, key=lambda x: x["cms"])
                if page["cms"] > victim["cms"]:
                    living.remove(victim)
                    living.append(page)
                    eviction_count += 1
                else:
                    rejected_count += 1

        # Adaptive capacity
        emergency_pages = sum(1 for p in living if p["cms"] >= emergency_threshold)
        adaptive_capacity = memory_size
        if emergency_pages > adaptive_trigger_count:
            adaptive_capacity += adaptive_expansion_step

        fifo_avg = sum(p["cms"] for p in fifo) / max(1, len(fifo))
        living_avg = sum(p["cms"] for p in living) / max(1, len(living))

        protected_pages = sum(1 for p in living if p["protected"])
        high_priority_pages = sum(1 for p in living if p["cms"] >= 0.60)
        fifo_protected = sum(1 for p in fifo if p["protected"])

        improvement_pct = 0.0
        if fifo_avg > 0:
            improvement_pct = round(((living_avg - fifo_avg) / fifo_avg) * 100.0, 2)

        return {
            "stream_size": len(scored_pages),
            "memory_size": memory_size,
            "adaptive_capacity": adaptive_capacity,
            "fifo_average_cms": round(fifo_avg, 4),
            "living_average_cms": round(living_avg, 4),
            "improvement_pct": improvement_pct,
            "protected_pages": protected_pages,
            "fifo_protected_pages": fifo_protected,
            "high_priority_pages": high_priority_pages,
            "emergency_pages": emergency_pages,
            "rejected_count": rejected_count,
            "eviction_count": eviction_count,
            "living_memory": sorted(living, key=lambda x: x["cms"], reverse=True),
            "fifo_memory": fifo,
        }

    def triage_single_patient(
        self,
        anchor_age: float,
        los: float,
        transfer_count: float,
        abnormal: float,
        admission_type: str = "EMERGENCY",
        first_careunit: str = "MICU",
        subject_id: str = "NEW_PATIENT",
        cms_weights: dict = None,
        protected_thresholds: dict = None,
    ) -> dict:
        age_risk = min(float(anchor_age) / 100.0, 1.0)
        icu_risk = min(float(los) / 10.0, 1.0)
        escalation_risk = min(float(transfer_count) / 10.0, 1.0)
        biomarker_risk = min(float(abnormal) / 20.0, 1.0)

        ai_risk = self.classifier.predict_ai_risk(
            age_risk=age_risk,
            icu_risk=icu_risk,
            escalation_risk=escalation_risk,
            biomarker_risk=biomarker_risk,
        )

        scorer = MimicClinicalScorer(
            cms_weights=cms_weights,
            protected_thresholds=protected_thresholds,
        )

        eval_res = scorer.evaluate_patient(
            ai_risk=ai_risk,
            age_risk=age_risk,
            icu_risk=icu_risk,
            escalation_risk=escalation_risk,
            biomarker_risk=biomarker_risk,
        )

        return {
            "subject_id": subject_id,
            "admission_type": admission_type,
            "first_careunit": first_careunit,
            "anchor_age": anchor_age,
            "los": los,
            "transfer_count": transfer_count,
            "abnormal": abnormal,
            "age_risk": round(age_risk, 4),
            "icu_risk": round(icu_risk, 4),
            "escalation_risk": round(escalation_risk, 4),
            "biomarker_risk": round(biomarker_risk, 4),
            "ai_risk": eval_res["ai_risk"],
            "inheritance": eval_res["inheritance"],
            "cms": eval_res["cms"],
            "protected": eval_res["protected"],
            "tier": eval_res["tier"],
            "reason": eval_res["reason"],
        }

