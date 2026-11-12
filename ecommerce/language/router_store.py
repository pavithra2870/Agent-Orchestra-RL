"""Build the router used by the backend (examples A + C, tuned parameters loaded from JSON)."""

from __future__ import annotations

import json
import threading
from typing import Optional

from ecommerce.config import RESULTS_DIR
from ecommerce.language.router import IntentRouter
from ecommerce.sim.text import example_sentences

PARAMS_PATH = RESULTS_DIR / "language" / "router_params.json"
_ROUTER: Optional[IntentRouter] = None
_LOCK = threading.Lock()


def get_router() -> IntentRouter:
    global _ROUTER
    with _LOCK:
        if _ROUTER is None:
            a, c = example_sentences("A"), example_sentences("C")
            router = IntentRouter(examples={k: a[k] + c[k] for k in a})
            if PARAMS_PATH.exists():
                router.load_params(json.loads(PARAMS_PATH.read_text()))
            _ROUTER = router
    return _ROUTER
