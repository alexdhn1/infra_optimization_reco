from typing import Dict, Any, Optional

class ThresholdProvider:
    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Return thresholds {warning, critical} for metric or None if not handled"""
        raise NotImplementedError
