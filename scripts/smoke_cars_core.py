"""Smoke test for CARS serializer, cache, appraiser, and reward shaper."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carsrl.cars import AppraisalCache, CARSRewardShaper, MiniGridStateSerializer, MockAppraiser
from carsrl.cars.serializer import EpisodeTrace


class _DummyObj:
    def __init__(self, type_: str, color: str):
        self.type = type_
        self.color = color


class _DummyUnwrapped:
    mission = "use the key to open the door and reach the goal"
    agent_pos = (2, 3)
    agent_dir = 0
    carrying = _DummyObj("key", "yellow")


class _DummyEnv:
    unwrapped = _DummyUnwrapped()


def _dummy_obs() -> dict[str, object]:
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[:, :, 0] = 1
    image[3, 4] = np.array([4, 4, 2], dtype=np.uint8)
    image[2, 5] = np.array([8, 0, 0], dtype=np.uint8)
    return {
        "mission": "use the key to open the door and reach the goal",
        "image": image,
    }


def main() -> None:
    trace = EpisodeTrace(history_length=4)
    trace.observe_transition(action=3, reward=0.0, terminated=False, truncated=False, info={"picked_up": "yellow key"})
    serializer = MiniGridStateSerializer(include_history=True, history_length=4)
    serialized = serializer.serialize(_DummyEnv(), _dummy_obs(), trace=trace)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = AppraisalCache(Path(tmpdir) / "appraisals.jsonl")
        appraiser = MockAppraiser(cache=cache, seed=7)
        previous = appraiser.appraise(serialized)
        current = appraiser.appraise(serialized + "\nRecent events: opened locked yellow door")
        shaper = CARSRewardShaper(beta=0.1, gamma=0.99)
        shaped = shaper.shape(previous, current)

        assert previous.phi >= 0.0
        assert current.confidence >= 0.0
        assert -0.05 <= shaped <= 0.05
        assert len(cache) == 2

    print("CARS core smoke test passed.")
    print("Serialized state:")
    print(serialized)
    print(f"Previous phi={previous.phi:.3f}, current phi={current.phi:.3f}, shaped={shaped:.4f}")


if __name__ == "__main__":
    main()
