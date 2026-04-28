from pathlib import Path

from carsrl.cars import Appraisal, AppraisalCache, CARSRewardShaper, MiniGridStateSerializer, MockAppraiser, ShuffledPhiAppraiser
from carsrl.cars.appraiser import _appraisal_validation_issue
from carsrl.cars.prompts import PROMPT_VERSION, build_appraisal_prompt
from carsrl.config import CARSConfig
from carsrl.ppo.model import obs_to_tensor


def test_appraisal_json_parse_clamps_values() -> None:
    appraisal = Appraisal.from_json_text('{"phi": 2, "confidence": -1, "subgoal": "x", "affordance": 0.5, "novelty": 0.2, "risk": 9}')
    assert appraisal.phi == 1.0
    assert appraisal.confidence == 0.0
    assert appraisal.risk == 1.0


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = AppraisalCache(tmp_path / "cache.jsonl")
    appraiser = MockAppraiser(cache=cache)
    first = appraiser.appraise("Mission: find key")
    second = appraiser.appraise("Mission: find key")
    assert first == second
    assert cache.stats.hits == 1


def test_potential_shaping_clips() -> None:
    previous = Appraisal(phi=0.0, confidence=1.0, subgoal="a", affordance=0.0, novelty=0.0, risk=0.0)
    current = Appraisal(phi=1.0, confidence=1.0, subgoal="b", affordance=0.0, novelty=0.0, risk=0.0)
    shaper = CARSRewardShaper(beta=1.0, gamma=0.99)
    assert shaper.shape(previous, current) == 0.05


def test_shuffled_phi_keeps_base_fields_but_reuses_previous_phi() -> None:
    base = MockAppraiser(seed=0)
    shuffled = ShuffledPhiAppraiser(base_appraiser=base, seed=123)

    first = shuffled.appraise("Mission: find key\nVisible objects: yellow key\nAgent inventory: carrying nothing")
    second = shuffled.appraise("Mission: reach goal\nVisible objects: goal\nAgent inventory: carrying yellow key")

    assert second.phi == first.phi
    assert second.confidence == 0.75
    assert second.subgoal.startswith("shuffled_phi:")


def test_qwen_prompt_does_not_embed_copyable_numeric_json() -> None:
    prompt = build_appraisal_prompt(
        "Mission: use the key to open the door and reach the goal\n"
        "Agent inventory: carrying yellow key\n"
        "Visible objects: locked yellow door at view(3,4)"
    )

    assert PROMPT_VERSION == "cars_appraisal_v6"
    assert '"phi": 0.25' not in prompt
    assert "pick up the visible key" not in prompt


def test_qwen_validation_rejects_legacy_copy_for_key_inventory() -> None:
    appraisal = Appraisal(
        phi=0.25,
        confidence=0.80,
        subgoal="pick up the visible key",
        affordance=0.70,
        novelty=0.30,
        risk=0.10,
    )
    state = (
        "Mission: use the key to open the door and reach the goal\n"
        "Agent inventory: carrying yellow key\n"
        "Visible objects: locked yellow door at view(3,4)"
    )

    assert _appraisal_validation_issue(appraisal, state) is not None


def test_qwen_validation_rejects_door_subgoal_when_key_visible() -> None:
    appraisal = Appraisal(
        phi=0.75,
        confidence=0.90,
        subgoal="toggle door",
        affordance=0.90,
        novelty=0.90,
        risk=0.10,
    )
    state = (
        "Mission: use the key to open the door and reach the goal\n"
        "Agent inventory: carrying nothing\n"
        "Visible objects: yellow key at view(2,5)"
    )

    assert _appraisal_validation_issue(appraisal, state) in {
        "subgoal_ignores_visible_key",
        "phi_too_high_for_visible_key_not_carried",
    }


def test_obs_to_tensor_includes_mission_and_direction_features() -> None:
    import numpy as np

    image = np.zeros((7, 7, 3), dtype=np.uint8)
    obs_a = {"image": image, "direction": 0, "mission": "pick up the yellow ball"}
    obs_b = {"image": image, "direction": 1, "mission": "pick up the green key"}

    tensor_a = obs_to_tensor(obs_a, "cpu")
    tensor_b = obs_to_tensor(obs_b, "cpu")

    assert tensor_a.shape == (1, 39, 7, 7)
    assert float((tensor_a - tensor_b).abs().sum()) > 0.0


def test_cars_defaults_are_markov_style_and_skip_neutral() -> None:
    config = CARSConfig()

    assert config.include_history is False
    assert config.neutral_on_skip is True


def test_serializer_history_flag_controls_last_action_and_recent_events() -> None:
    import numpy as np

    obs = {"image": np.zeros((7, 7, 3), dtype=np.uint8), "mission": "reach the goal", "direction": 0}
    state_only = MiniGridStateSerializer(include_history=False).serialize(
        None,
        obs,
        last_action=2,
        recent_events=["picked up key"],
        action_history=[2],
    )
    with_history = MiniGridStateSerializer(include_history=True).serialize(
        None,
        obs,
        last_action=2,
        recent_events=["picked up key"],
        action_history=[2],
    )

    assert "Last action" not in state_only
    assert "Recent events" not in state_only
    assert "Last action: move forward" in with_history
    assert "Recent events: picked up key" in with_history
