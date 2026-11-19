"""Unit tests to validate the Exclusive 10-Fold CV Scheduler and Sequential Heavy Model policy."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.training.adaptive_parallel_scheduler import (
    AdaptiveParallelScheduler,
    JobSpec,
    RunningJob,
    MODEL_PROFILES,
    HEAVY_MODELS,
)


class TestSequentialHeavyScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = AdaptiveParallelScheduler(
            n_folds_list=[10],
            max_concurrency=4,
            sequential_heavy=True,
        )

    def test_n_folds_strictly_10(self):
        """Scheduler must only have 10-fold in its regime list."""
        self.assertEqual(self.scheduler.n_folds_list, [10])

    def test_heavy_models_classification(self):
        """Verify all 6 heavy architectures are categorized as heavy."""
        expected_heavy = {
            "resnet50", "densenet121", "efficientnet_b0",
            "resnet50_lstm", "resnet50_bilstm", "efficientnet_b0_bilstm"
        }
        self.assertEqual(HEAVY_MODELS, expected_heavy)
        for m in expected_heavy:
            self.assertTrue(MODEL_PROFILES[m]["is_heavy"])

    @patch.object(AdaptiveParallelScheduler, "get_hardware_telemetry")
    def test_sequential_heavy_rejection_when_one_running(self, mock_telemetry):
        """When 1 heavy model is already running, reject another heavy model."""
        mock_telemetry.return_value = {
            "vram_pct": 50.0,
            "available_ram_gb": 8.0,
            "swap_pct": 10.0,
            "cpu_pct": 30.0,
        }
        # Simulate 1 active heavy worker
        dummy_proc = MagicMock()
        h1 = RunningJob(
            spec=JobSpec("fundus_3class", "efficientnet_b0_bilstm", 9, 10, "cnn_hybrid", True, 32, 1250.0),
            process=dummy_proc, start_time=0.0, log_path=Path("dummy.log")
        )
        self.scheduler.running = [h1]

        # Candidate 2: another heavy job
        h2_spec = JobSpec("oct_individual", "resnet50", 1, 10, "cnn_standalone", True, 32, 1800.0)
        eligible, reason = self.scheduler.can_dispatch(h2_spec)
        self.assertFalse(eligible)
        self.assertIn("Sequential heavy policy", reason)

    @patch.object(AdaptiveParallelScheduler, "get_hardware_telemetry")
    def test_sequential_heavy_lock_same_model(self, mock_telemetry):
        """If heavy target lock is set to (fundus, efficientnet_b0_bilstm), another heavy model is rejected."""
        mock_telemetry.return_value = {
            "vram_pct": 50.0,
            "available_ram_gb": 8.0,
            "swap_pct": 10.0,
            "cpu_pct": 30.0,
        }
        self.scheduler.running = []  # No jobs running currently
        self.scheduler.current_heavy_target = ("fundus_3class", "efficientnet_b0_bilstm")
        # Add remaining fold of current target to queue
        self.scheduler.queue = [
            JobSpec("fundus_3class", "efficientnet_b0_bilstm", 10, 10, "cnn_hybrid", True, 32, 1250.0)
        ]

        # Candidate from different heavy model
        other_heavy = JobSpec("oct_severity", "resnet50", 1, 10, "cnn_standalone", True, 32, 1800.0)
        eligible, reason = self.scheduler.can_dispatch(other_heavy)
        self.assertFalse(eligible)
        self.assertIn("Sequential heavy lock", reason)

        # Candidate from current heavy target
        same_heavy = JobSpec("fundus_3class", "efficientnet_b0_bilstm", 10, 10, "cnn_hybrid", True, 32, 1250.0)
        eligible_same, _ = self.scheduler.can_dispatch(same_heavy)
        self.assertTrue(eligible_same)


if __name__ == "__main__":
    unittest.main()
