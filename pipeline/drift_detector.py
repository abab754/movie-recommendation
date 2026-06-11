"""Detects rating distribution drift over time."""

import numpy as np


class DriftDetector:
    """Monitors rating distribution and flags drift when current window
    deviates >2 stddev from baseline."""

    def __init__(self):
        self.baseline_mean: float | None = None
        self.baseline_std: float | None = None
        self.baseline_ratings: list[float] = []
        self.baseline_set = False
        self.baseline_size = 100_000  # first 100K events form the baseline
        self.window: list[float] = []
        self.window_size = 1000

    def add_rating(self, rating: float) -> list[dict] | None:
        """Add a rating and return drift results if window is full."""
        # Build baseline from first N ratings
        if not self.baseline_set:
            self.baseline_ratings.append(rating)
            if len(self.baseline_ratings) >= self.baseline_size:
                arr = np.array(self.baseline_ratings)
                self.baseline_mean = float(np.mean(arr))
                self.baseline_std = float(np.std(arr))
                self.baseline_set = True
                self.baseline_ratings = []  # free memory
                print(
                    f"Drift baseline set: mean={self.baseline_mean:.3f}, "
                    f"std={self.baseline_std:.3f}"
                )
            return None

        self.window.append(rating)

        if len(self.window) >= self.window_size:
            results = self._check_drift()
            self.window = []
            return results

        return None

    def _check_drift(self) -> list[dict]:
        arr = np.array(self.window)
        current_mean = float(np.mean(arr))
        current_std = float(np.std(arr))

        results = []

        # Check mean drift
        mean_drift = abs(current_mean - self.baseline_mean) > 2 * self.baseline_std
        results.append(
            {
                "metric": "rating_mean",
                "baseline_value": self.baseline_mean,
                "current_value": current_mean,
                "drift_detected": mean_drift,
            }
        )

        # Check stddev drift
        std_drift = abs(current_std - self.baseline_std) > 2 * self.baseline_std
        results.append(
            {
                "metric": "rating_stddev",
                "baseline_value": self.baseline_std,
                "current_value": current_std,
                "drift_detected": std_drift,
            }
        )

        if mean_drift or std_drift:
            print(
                f"DRIFT DETECTED: mean={current_mean:.3f} "
                f"(baseline={self.baseline_mean:.3f}), "
                f"std={current_std:.3f} (baseline={self.baseline_std:.3f})"
            )

        return results
