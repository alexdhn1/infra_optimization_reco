# thresholds.py
from __future__ import annotations
import yaml
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from bisect import bisect_left

Threshold = Dict[str, float]  # {"warning": float, "critical": float}

@dataclass
class HysteresisConfig:
    consecutive_points: int = 2
    lookback_points: int = 5

class ThresholdProvider:
    """Interface for threshold providers."""
    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Threshold]:
        raise NotImplementedError

class StaticThresholdProvider(ThresholdProvider):
    def __init__(self, table: Dict[str, Threshold], overrides: Optional[Dict[str, Dict[str, Threshold]]] = None):
        self.table = table or {}
        self.overrides = overrides or {}

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Threshold]:
        svc = context.get("service")
        if svc and svc in self.overrides and metric in self.overrides[svc]:
            return self.overrides[svc][metric]
        return self.table.get(metric)

class SLOThresholdProvider(ThresholdProvider):
    """
    Derives thresholds from SLO objectives.
    Example: latency p95 objective 200ms -> warning at 1.1*200=220, critical at 1.25*200=250
    """
    def __init__(self, slos: Dict[str, Any]):
        self.slos = slos or {}

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Threshold]:
        cfg = self.slos.get(metric)
        if not cfg:
            return None
        if metric == "latency_ms":
            base = float(cfg.get("objective_ms_p95", 0))
        else:
            base = float(cfg.get("objective", 0))
        if base <= 0:
            return None
        warn = base * float(cfg.get("warn_factor", 1.0))
        crit = base * float(cfg.get("crit_factor", 1.5))
        # ensure warning <= critical for higher-is-worse metrics
        warn, crit = min(warn, crit), max(warn, crit)
        return {"warning": warn, "critical": crit}

class PercentileThresholdProvider(ThresholdProvider):
    """
    Computes thresholds from history percentiles (e.g., P90/P99) per metric and service.
    History format: List[Dict] where each item has metric fields and optional 'timestamp' and 'service'.
    """
    def __init__(self, history: List[Dict[str, Any]], p_warn: float = 0.90, p_crit: float = 0.99):
        self.p_warn = p_warn
        self.p_crit = p_crit
        self.history = history or []
        # Pre-index by service for fast lookup
        self.by_service: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for row in self.history:
            svc = row.get("service")  # optional; user can inject service into each record
            self.by_service.setdefault(svc, []).append(row)
        # Optionally sort by timestamp if present
        for svc, arr in self.by_service.items():
            arr.sort(key=lambda d: d.get("timestamp", ""))

    @staticmethod
    def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
        if not sorted_vals:
            return None
        idx = max(0, min(len(sorted_vals) - 1, int(q * len(sorted_vals)) - 1))
        return sorted_vals[idx]

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Threshold]:
        svc = context.get("service")
        arr = self.by_service.get(svc, []) or self.by_service.get(None, [])
        vals = [float(x[metric]) for x in arr if metric in x and isinstance(x[metric], (int, float))]
        if not vals:
            return None
        vals.sort()
        warn = self._percentile(vals, self.p_warn)
        crit = self._percentile(vals, self.p_crit)
        if warn is None or crit is None:
            return None
        # Ensure warn <= crit
        warn, crit = min(warn, crit), max(warn, crit)
        return {"warning": warn, "critical": crit}

    # --- helpers for hysteresis ---
    def consecutive_breaches(
        self,
        metric: str,
        threshold: float,
        current_ts: Optional[str],
        service: Optional[str],
        lookback_points: int,
    ) -> int:
        """Count consecutive breaches up to current_ts within lookback window."""
        arr = self.by_service.get(service, []) or self.by_service.get(None, [])
        if not arr:
            return 0
        # find position of current_ts
        if current_ts is not None:
            ts_list = [row.get("timestamp", "") for row in arr]
            pos = bisect_left(ts_list, current_ts)
            # include current index if exact match; otherwise look at the last few before the insert pos and include current snapshot as last item
            end = min(len(arr), pos + 1)
        else:
            end = len(arr)
        window = arr[max(0, end - lookback_points): end]
        cnt = 0
        for row in reversed(window):
            v = row.get(metric)
            if isinstance(v, (int, float)) and float(v) >= threshold:
                cnt += 1
            else:
                break
        return cnt

class CompositeThresholdProvider(ThresholdProvider):
    """
    Combines multiple providers. By default, picks the 'stricter' (lower) thresholds for higher-is-worse metrics.
    Order: earlier providers have precedence if 'prefer_first=True'; otherwise choose stricter.
    """
    def __init__(self, providers: List[ThresholdProvider], prefer_first: bool = False):
        self.providers = providers
        self.prefer_first = prefer_first

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Threshold]:
        chosen: Optional[Threshold] = None
        for p in self.providers:
            th = p.get(metric, context)
            if not th:
                continue
            if self.prefer_first and chosen is None:
                chosen = th
                continue
            if chosen is None:
                chosen = th
            else:
                # stricter for higher-is-worse metrics => lower thresholds
                chosen = {
                    "warning": min(chosen["warning"], th["warning"]),
                    "critical": min(chosen["critical"], th["critical"]),
                }
        return chosen

def load_threshold_stack(
    yaml_path: str,
    history: List[Dict[str, Any]],
) -> Tuple[ThresholdProvider, HysteresisConfig]:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    static_table = (cfg.get("static") or {})
    overrides = (cfg.get("overrides") or {})
    slos = (cfg.get("slos") or {})
    hyst = cfg.get("hysteresis") or {}

    static_provider = StaticThresholdProvider(static_table, overrides)
    slo_provider = SLOThresholdProvider(slos)
    pct_provider = PercentileThresholdProvider(history)

    provider = CompositeThresholdProvider(
        providers=[static_provider, slo_provider, pct_provider],
        prefer_first=False,  # choose stricter among them
    )
    hcfg = HysteresisConfig(
        consecutive_points=int(hyst.get("consecutive_points", 2)),
        lookback_points=int(hyst.get("lookback_points", 5)),
    )
    return provider, hcfg
