from __future__ import annotations

import unittest

from local_probability_server import DetectionStore


class DetectionStoreTest(unittest.TestCase):
    def test_confirmed_version_keeps_increasing_after_clear(self) -> None:
        store = DetectionStore()
        first = store.put(
            {
                "before": {"white": [0.1, 0.2]},
                "shooter": "white",
                "prediction": {"successProbability": 0.4},
                "analysis": {"confirmed": True},
            }
        )
        self.assertEqual(first["confirmedVersion"], 1)

        cleared = store.clear()
        self.assertNotIn("confirmedVersion", cleared)
        self.assertNotIn("confirmedVersion", store.get())

        second = store.put(
            {
                "before": {"white": [0.2, 0.3]},
                "shooter": "yellow",
                "prediction": {"successProbability": 0.5},
                "analysis": {"confirmed": True},
            }
        )
        self.assertEqual(second["confirmedVersion"], 2)


if __name__ == "__main__":
    unittest.main()
