"""
MIMIC-IV Clinical Priority Scoring Engine

This module implements the core decision-making logic of LivingMemoryOS by
transforming patient risk indicators into actionable clinical priority scores.
It calculates Patient-Criticality Inheritance, derives the composite Clinical
Memory Score (CMS), determines prognostic retention protection status, and
assigns patients to appropriate memory management tiers based on severity and
clinical urgency. The scoring framework integrates AI-predicted mortality
risk, biomarker abnormalities, ICU burden, escalation patterns, and age-related
risk factors to support intelligent retention, prioritization, and adaptive
resource allocation within the LivingMemoryOS architecture.
"""

from src.config_new import (
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_PROTECTED_THRESHOLDS,
    EMERGENCY_CMS_THRESHOLD,
)


class MimicClinicalScorer:
    """Computes multidimensional clinical priority scores for the MIMIC-IV dataset."""

    def __init__(
        self,
        cms_weights: dict = None,
        protected_thresholds: dict = None,
        emergency_threshold: float = EMERGENCY_CMS_THRESHOLD,
    ):
        self.cms_weights = cms_weights or DEFAULT_CMS_WEIGHTS
        self.protected_thresholds = protected_thresholds or DEFAULT_PROTECTED_THRESHOLDS
        self.emergency_threshold = emergency_threshold

    def compute_inheritance(self, icu_risk: float, escalation_risk: float) -> float:
        """Calculates Patient-Criticality Inheritance: 0.5 * icu_risk + 0.5 * escalation_risk."""
        return float(round(0.5 * icu_risk + 0.5 * escalation_risk, 4))

    def compute_cms(
        self,
        ai_risk: float,
        biomarker_risk: float,
        escalation_risk: float,
        icu_risk: float,
        age_risk: float,
        inheritance: float = None,
    ) -> float:
        """Calculates composite Clinical Memory Score (CMS)."""
        if inheritance is None:
            inheritance = self.compute_inheritance(icu_risk, escalation_risk)

        w = self.cms_weights
        cms = (
            w.get("ai_risk", 0.30) * ai_risk
            + w.get("biomarker_risk", 0.20) * biomarker_risk
            + w.get("escalation_risk", 0.20) * escalation_risk
            + w.get("icu_risk", 0.15) * icu_risk
            + w.get("age_risk", 0.10) * age_risk
            + w.get("inheritance", 0.05) * inheritance
        )
        return float(round(cms, 4))

    def is_protected(self, ai_risk: float, biomarker_risk: float) -> bool:
        """Determines prognostic retention protection: (ai_risk > 0.75) | (biomarker_risk > 0.80)."""
        ai_cutoff = self.protected_thresholds.get("ai_risk_cutoff", 0.75)
        bio_cutoff = self.protected_thresholds.get("biomarker_risk_cutoff", 0.80)
        return bool((ai_risk > ai_cutoff) or (biomarker_risk > bio_cutoff))

    def evaluate_patient(
        self,
        ai_risk: float,
        age_risk: float,
        icu_risk: float,
        escalation_risk: float,
        biomarker_risk: float,
    ) -> dict:
        """Evaluates patient and returns scores, protection flag, tier, and clinical rationale."""
        inheritance = self.compute_inheritance(icu_risk, escalation_risk)
        cms = self.compute_cms(
            ai_risk=ai_risk,
            biomarker_risk=biomarker_risk,
            escalation_risk=escalation_risk,
            icu_risk=icu_risk,
            age_risk=age_risk,
            inheritance=inheritance,
        )
        protected = self.is_protected(ai_risk, biomarker_risk)

        if protected or cms >= self.emergency_threshold:
            tier = "EMERGENCY"
        elif cms >= 0.60:
            tier = "HIGH_PRIORITY"
        else:
            tier = "NORMAL"

        reasons = []
        if protected:
            reasons.append("Prognostic Non-Evictable Protection")
        if ai_risk > self.protected_thresholds.get("ai_risk_cutoff", 0.75):
            reasons.append("Critical AI Mortality Risk")
        if biomarker_risk > self.protected_thresholds.get("biomarker_risk_cutoff", 0.80):
            reasons.append("Severe Lab Biomarker Abnormalities")
        if escalation_risk >= 0.50:
            reasons.append("High Transfer Escalation")
        if icu_risk >= 0.50:
            reasons.append("Prolonged ICU Stay")
        if age_risk >= 0.70:
            reasons.append("Geriatric High Risk")
        if not reasons:
            reasons.append("Routine Inpatient Care")

        return {
            "ai_risk": ai_risk,
            "inheritance": inheritance,
            "cms": cms,
            "protected": protected,
            "tier": tier,
            "reason": ", ".join(reasons),
        }

