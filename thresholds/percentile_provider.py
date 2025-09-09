import statistics
from typing import Dict, Any, Optional, List
from .base import ThresholdProvider

class PercentileThresholdProvider(ThresholdProvider):
    def __init__(self, history: List[Dict[str, Any]], p_warn=0.9, p_crit=0.99):
        self.history = history
        self.p_warn = p_warn
        self.p_crit = p_crit

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        series = [d[metric] for d in self.history if metric in d]
        if not series: return None
        series.sort()
        n = len(series)
        warn = series[int(self.p_warn * n) - 1]
        crit = series[int(self.p_crit * n) - 1]
        return {"warning": warn, "critical": crit}
