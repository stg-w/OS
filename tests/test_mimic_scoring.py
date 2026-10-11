import unittest

from src.models.mimic_scoring import MimicClinicalScorer


class MimicClinicalScorerTests(unittest.TestCase):
    def test_inheritance_is_the_mean_of_icu_and_escalation_risk(self):
        scorer = MimicClinicalScorer()

        self.assertEqual(scorer.compute_inheritance(0.8, 0.4), 0.6)

    def test_protected_patient_is_assigned_to_emergency(self):
        scorer = MimicClinicalScorer()

        result = scorer.evaluate_patient(
            ai_risk=0.9,
            age_risk=0.3,
            icu_risk=0.2,
            escalation_risk=0.2,
            biomarker_risk=0.1,
        )

        self.assertTrue(result["protected"])
        self.assertEqual(result["tier"], "EMERGENCY")


if __name__ == "__main__":
    unittest.main()
