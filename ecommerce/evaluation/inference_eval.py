"""How often does order inference pick the right order, ask a short question, or give up?

Setup (synthetic, so read the caveat): for every customer with 2+ orders in 40 generated worlds and every goal, the
customer says "my order" with no id. The INTENDED order is drawn at random from the orders that qualify for the goal
(a person asking for a refund has an order that can be refunded in mind). For "where is it" the intended order is any
order still on its way, or a lost / just-delivered one when nothing is in transit.

Outcomes: right (inferred the intended order), wrong (inferred another one), asked (several candidates, list narrowed),
none (no order qualifies, so the agent asks and says why), and the average size of the list the customer is shown
compared with all their orders.

Caveat: the intended order is chosen from the same rules the inference uses, so "wrong" can only come from the
delivery fallback. Real customers also ask about orders that do not qualify; those cases end in "none" or a question,
never in a silent state change, because assumed orders for refund / cancel / payment change need a yes first.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from ecommerce.config import get_settings
from ecommerce.language.orders import _qualifies, _recent_or_lost, resolve_order
from ecommerce.sim.world import generate_world

GOALS = ("delivery_status", "refund", "cancel_order", "payment_change")


def run(seeds: int = 40) -> dict:
    cfg = get_settings().world
    rng = np.random.default_rng(0)
    tally = defaultdict(Counter)
    shown, total_orders = defaultdict(list), defaultdict(list)
    for seed in range(seeds):
        w = generate_world(seed)
        today = date.fromisoformat(w.today)
        for c in w.customers.values():
            if len(c.order_ids) < 2:
                continue
            orders = [w.orders[i] for i in c.order_ids]
            for g in GOALS:
                target = "UPI" if g == "payment_change" else None
                intended = [o for o in orders if _qualifies(g, o, target, cfg, today)]
                if g == "delivery_status" and not intended:
                    intended = [o for o in orders if _recent_or_lost(o, today)]
                r = resolve_order("my order", [g], target, orders, today, cfg)
                if not intended:
                    tally[g]["no_intended_order"] += 1
                    tally[g]["asked_or_none" if r.how != "inferred" else "inferred_without_intended"] += 1
                    continue
                pick = intended[int(rng.integers(len(intended)))]
                if r.how == "inferred":
                    tally[g]["right" if r.order_id == pick.order_id else "wrong"] += 1
                elif r.how == "ambiguous":
                    tally[g]["asked"] += 1
                    shown[g].append(len(r.candidates))
                    total_orders[g].append(len(orders))
                else:
                    tally[g][r.how] += 1
                tally[g]["cases"] += 1
    out = {}
    for g in GOALS:
        t = tally[g]
        n = t["cases"] or 1
        out[g] = {"cases": t["cases"], "right": t["right"], "wrong": t["wrong"], "asked": t["asked"], "none_qualifies": t["none_qualifies"],
                  "right_rate": round(t["right"] / n, 3), "wrong_rate": round(t["wrong"] / n, 3), "asked_rate": round(t["asked"] / n, 3),
                  "avg_list_shown": round(float(np.mean(shown[g])), 2) if shown[g] else None,
                  "avg_orders_owned_when_asked": round(float(np.mean(total_orders[g])), 2) if total_orders[g] else None,
                  "no_qualifying_order_cases": t["no_intended_order"], "inferred_without_intended": t["inferred_without_intended"]}
    return out


if __name__ == "__main__":
    res = run()
    path = Path(__file__).resolve().parents[1] / "results" / "language" / "inference_eval.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
