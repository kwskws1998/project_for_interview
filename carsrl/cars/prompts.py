"""Prompt templates for frozen SLM appraisal."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a reward-shaping appraiser for MiniGrid reinforcement learning.
Estimate whether the current symbolic state shows progress toward the mission.
Return strict JSON only. Do not include markdown or commentary.
Infer values from the given state. Never reuse a generic key-door example when it conflicts with the
agent inventory, visible objects, or mission."""


PROMPT_VERSION = "cars_appraisal_v6"


def build_appraisal_prompt(serialized_state: str) -> str:
    return f"""Given the MiniGrid state below, produce a cognitive appraisal.

Definitions:
- phi: goal-progress potential from 0.0 to 1.0.
- confidence: confidence in the appraisal from 0.0 to 1.0.
- subgoal: the next meaningful subgoal.
- affordance: how actionable the next useful step appears from 0.0 to 1.0.
- novelty: whether the state is novel from 0.0 to 1.0.
- risk: risk of wasting time, being blocked, or moving away from the goal from 0.0 to 1.0.

Progress rubric for key-door-goal missions:
- 0.05 to 0.15: no useful object or route is visible.
- 0.20 to 0.35: key, door, or other useful object is visible but not yet used.
- 0.40 to 0.60: agent is carrying the key or is positioned to use an important object.
- 0.60 to 0.80: locked door is open or the agent has clear access to the goal area.
- 0.80 to 1.00: goal is visible/reachable or mission is nearly complete.

Consistency rules:
- If the agent is already carrying a key, the subgoal must not be to pick up a key.
- If a key is visible and the agent is not carrying a key, the subgoal should be to pick up that key.
- If a locked door is visible and the agent is carrying the matching key, the subgoal should be about opening/toggling the door.
- If the goal is visible and no locked door still blocks it, the subgoal should be to reach the goal, not to toggle/open a door.
- If the goal is visible or the route to the goal is open, phi should usually be high.
- If a locked door blocks progress and no key is visible or carried, the subgoal should be to find the matching key.
- If the visible useful object color/type contradicts the mission, lower affordance and explain the correct subgoal.
- The subgoal describes the next useful action, not the previous action.

State:
{serialized_state}

Return exactly one minified JSON object with exactly these keys:
phi, confidence, subgoal, affordance, novelty, risk.
The five numeric values must be JSON numbers in [0, 1].
The subgoal must be a short string describing the next useful action.
The first character of your answer must be {{ and the last character must be }}."""
