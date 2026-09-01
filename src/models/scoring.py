"""
Clinical Scoring Module
Calculates Clinical Memory Score (CMS), Care Escalation levels, and Triage Tiers.
"""

from src.config import (
    DEFAULT_CARE_LEVEL_THRESHOLDS,
    BED_PRIORITY,
    ESCALATION_SCORE,
    DEFAULT_CMS_WEIGHTS,
    DEFAULT_CRITICAL_THRESHOLD,
)


class ClinicalScorer:
    """Computes multidimensional clinical priority scores and triage allocations."""

    def __init__(
        self,
        care_thresholds: dict = None,
        cms_weights: dict = None,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    ):
        self.care_thresholds = care_thresholds or DEFAULT_CARE_LEVEL_THRESHOLDS
        self.cms_weights = cms_weights or DEFAULT_CMS_WEIGHTS
        self.critical_threshold = critical_threshold

    def assign_care_level(self, criticality: float) -> str:
        """Determines clinical care level (ER, ICU, HDU, WARD) based on calibrated risk."""
        if criticality >= self.care_thresholds.get("ER", 0.45):
            return "ER"
        if criticality >= self.care_thresholds.get("ICU", 0.35):
            return "ICU"
        if criticality >= self.care_thresholds.get("HDU", 0.20):
            return "HDU"
        return "WARD"

    def compute_cms(
        self,
        criticality: float,
        vitals: dict,
        care_level: str,
    ) -> tuple:
        """
        Computes composite Clinical Memory Score (CMS).
        Returns: (cms_score, severity_component, biomarker_risk_component)
        """
        apache = float(vitals.get("apache_score", 0.0))
        sofa = float(vitals.get("sofa_score", 0.0))
        lactate = float(vitals.get("lactate_mean", 0.0))
        sepsis = float(vitals.get("sepsis_flag", 0.0))

        # Normalized clinical risk components
        severity = min(((apache / 40.0) + (sofa / 20.0)) / 2.0, 1.0)
        biomarker_risk = min(lactate / 10.0, 1.0)
        sepsis_risk = float(sepsis)

        w = self.cms_weights
        cms = (
            w.get("criticality", 0.30) * criticality
            + w.get("severity", 0.20) * severity
            + w.get("bed_priority", 0.15) * BED_PRIORITY.get(care_level, 0.20)
            + w.get("biomarker_risk", 0.15) * biomarker_risk
            + w.get("sepsis_risk", 0.10) * sepsis_risk
            + w.get("escalation_score", 0.10) * ESCALATION_SCORE.get(care_level, 0.25)
        )
        return float(cms), float(severity), float(biomarker_risk)

    def determine_tier_and_reasons(
        self,
        criticality: float,
        vitals: dict,
        care_level: str,
    ) -> tuple:
        """
        Assigns memory tier (EMERGENCY, HIGH_PRIORITY, NORMAL) and generates human-readable clinical rationale.
        """
        sepsis_flag = int(vitals.get("sepsis_flag", 0))
        lactate = float(vitals.get("lactate_mean", 0.0))

        tier = "NORMAL"
        if criticality >= self.critical_threshold:
            tier = "HIGH_PRIORITY"
        if criticality >= self.critical_threshold and sepsis_flag == 1:
            tier = "EMERGENCY"

        reasons = []
        if criticality >= self.critical_threshold:
            reasons.append("High Mortality Risk")
        if sepsis_flag == 1:
            reasons.append("Sepsis")
        if lactate > 4.0:
            reasons.append("High Lactate")
        if care_level == "ER":
            reasons.append("ER Escalation")
        if tier == "EMERGENCY":
            reasons.append("Emergency Preservation")
        if not reasons:
            reasons.append("Low Clinical Priority")

        return tier, ", ".join(reasons)
