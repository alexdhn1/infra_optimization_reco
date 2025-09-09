from typing import Dict, Any, Optional, List
from .base import ThresholdProvider

class CompositeThresholdProvider(ThresholdProvider):
    def __init__(self, providers: List[ThresholdProvider]):
        self.providers = providers

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        chosen = None
        for p in self.providers:
            th = p.get(metric, context)
            if not th: continue
            if chosen is None:
                chosen = th
            else:
                chosen = {
                    "warning": min(chosen["warning"], th["warning"]),
                    "critical": min(chosen["critical"], th["critical"])
                }
        return chosen
