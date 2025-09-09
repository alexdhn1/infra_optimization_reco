from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .base import ThresholdProvider

class SLOBurnRateProvider(ThresholdProvider):
    def __init__(self, slos: Dict[str, Any], history: List[Dict[str, Any]]):
        self.slos = slos
        self.history = history

    def _burn_rate(self, values, objective):
        if not values: return 0
        breaches = sum(1 for v in values if v > objective)
        return breaches / len(values)

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        cfg = self.slos.get(metric)
        if not cfg:
            return None
        now = datetime.utcnow()
        short_window = timedelta(seconds=cfg["short_window"])
        long_window = timedelta(seconds=cfg["long_window"])
        values_short = [d[metric] for d in self.history if metric in d and now - datetime.fromisoformat(d["timestamp"].replace("Z","")) < short_window]
        values_long  = [d[metric] for d in self.history if metric in d and now - datetime.fromisoformat(d["timestamp"].replace("Z","")) < long_window]

        burn_short = self._burn_rate(values_short, cfg.get("objective_ms_p95", cfg.get("objective")))
        burn_long  = self._burn_rate(values_long,  cfg.get("objective_ms_p95", cfg.get("objective")))

        if burn_short > cfg["short_burn"] and burn_long > cfg["long_burn"]:
            obj = cfg.get("objective_ms_p95", cfg.get("objective"))
            return {"warning": obj, "critical": obj * 1.2}
        return None
