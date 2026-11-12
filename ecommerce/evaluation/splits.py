"""Test sets. Each split has its own seed range, so training and test scenarios never overlap.

  train           what the learning agents train on (6 goal combinations, phrasing set A)
  iid             same distribution as training, fresh seeds
  heldout_combos  goal combinations never seen in training
  tool_failures   3x more tool failures and sometimes a tool is fully down
  missing_info    missing / wrong order ids and low parser confidence (must ask)
  out_of_scope    requests the shop's tools cannot handle (must hand to a human)
  returns         return / exchange / damage-claim requests (all five outcomes: instant refund, return opened, exchange, human, declined)
  policy          policy questions, alone or together with a request (the agent must look the policy up; no order needed)
  paraphrase      wording from the held-out phrase set B: the final test of the real parser pipeline
  language_val    wording from phrase set C: only used to tune / calibrate the router
"""

from __future__ import annotations

from typing import Dict, Optional

from ecommerce.config import Settings, get_settings
from ecommerce.sim.scenarios import ALL_COMBOS, HELDOUT_COMBOS, TRAIN_COMBOS, SplitSpec

# Group B test sets (the held-out goal combinations are ('delivery_status', 'return_item') and ('policy_info', 'delivery_status'))
RETURN_COMBOS = [("return_item",), ("delivery_status", "return_item"), ("policy_info", "return_item")]
POLICY_COMBOS = [("policy_info",), ("policy_info", "refund"), ("policy_info", "cancel_order"), ("policy_info", "delivery_status")]

EVAL_SIZES = {
    "iid": 300, "heldout_combos": 200, "tool_failures": 200, "missing_info": 150,
    "out_of_scope": 60, "paraphrase": 200, "returns": 300, "policy": 200,
}


def get_splits(settings: Optional[Settings] = None) -> Dict[str, SplitSpec]:
    settings = settings or get_settings()
    sc = settings.scenarios
    standard_only = {"standard": 1.0}
    return {
        "train": SplitSpec("train", 1000, TRAIN_COMBOS),
        "validation": SplitSpec("validation", 2000, TRAIN_COMBOS),
        "iid": SplitSpec("iid", 3000, TRAIN_COMBOS),
        "heldout_combos": SplitSpec("heldout_combos", 4000, HELDOUT_COMBOS),
        "paraphrase": SplitSpec("paraphrase", 5000, ALL_COMBOS, phrasing_set="B"),
        "language_val": SplitSpec("language_val", 5500, ALL_COMBOS, phrasing_set="C"),
        "tool_failures": SplitSpec("tool_failures", 6000, TRAIN_COMBOS, type_probs=standard_only,
                                   failure_multiplier=sc.failure_multiplier_stress, down_prob=sc.down_tool_prob_stress),
        "missing_info": SplitSpec("missing_info", 7000, TRAIN_COMBOS,
                                  type_probs={"missing_order": 0.4, "wrong_order": 0.2, "low_confidence": 0.4}),
        "out_of_scope": SplitSpec("out_of_scope", 8000, TRAIN_COMBOS, type_probs={"out_of_scope": 1.0}),
        "returns": SplitSpec("returns", 9000, RETURN_COMBOS, type_probs=standard_only),
        "policy": SplitSpec("policy", 9500, POLICY_COMBOS, type_probs=standard_only),
    }
