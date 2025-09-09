import statistics
from typing import Dict, Any, Optional, List
from .base import ThresholdProvider

class AdaptiveStatsProvider(ThresholdProvider):
    def __init__(self, history: List[Dict[str, Any]], alpha=0.25):
        self.history = history
        self.alpha = alpha

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        series = [d[metric] for d in self.history if metric in d]
        if len(series) < 10: return None
        mu, sigma = statistics.mean(series), statistics.stdev(series)
        ewma = series[0]
        for v in series[1:]:
            ewma = self.alpha*v + (1-self.alpha)*ewma
        sigma_ewma = statistics.stdev(series)
        warn = max(mu + 2*sigma, ewma + 2*sigma_ewma)
        crit = max(mu + 3*sigma, ewma + 3*sigma_ewma)
        return {"warning": warn, "critical": crit}
