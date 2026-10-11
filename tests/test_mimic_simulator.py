import unittest

from src.memory.mimic_simulator import MimicLivingMemorySimulator


class _FakeClassifier:
    is_trained = True


class MimicSimulatorTests(unittest.TestCase):
    def test_adaptive_capacity_expands_during_ingestion(self):
        simulator = MimicLivingMemorySimulator(classifier=_FakeClassifier())
        simulator.load_stream_pages = lambda max_pages: [
            {
                "ai_risk": 0.9,
                "age_risk": 0.8,
                "icu_risk": 0.8,
                "escalation_risk": 0.8,
                "biomarker_risk": 0.9,
            }
            for _ in range(4)
        ]

        result = simulator.run_simulation(
            memory_size=2,
            stream_size=4,
            adaptive_trigger_count=1,
            adaptive_expansion_step=1,
        )

        self.assertGreater(result["adaptive_capacity"], 2)
        self.assertEqual(len(result["living_memory"]), 4)
        self.assertEqual(result["rejected_count"], 0)


if __name__ == "__main__":
    unittest.main()
