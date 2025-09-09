# nodes/anomaly.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime
from thresholds.base import ThresholdProvider
from nodes.ingestion import MetricsData, ServiceStatus

__all__ = ["AnomalyDetectionNode"]

class AnomalyDetectionNode:
    def __init__(
        self,
        provider: ThresholdProvider,
        history: Optional[List[Dict[str, Any]]] = None,
        consecutive_points: int = 3,
        lookback_points: int = 5,
    ):
        self.provider = provider
        self.history = history or []
        self.consecutive_points = max(1, consecutive_points)
        self.lookback_points = max(self.consecutive_points, lookback_points)

    @staticmethod
    def _to_dt(ts: Optional[str]) -> Optional[datetime]:
        if not ts: return None
        try: return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception: return None

    def _consecutive_breaches(self, metric: str, threshold: float, current_ts: Optional[str]) -> int:
        if not self.history: return 1
        dt_cur = self._to_dt(current_ts)
        rows = []
        if dt_cur:
            for d in self.history:
                ts = self._to_dt(d.get("timestamp"))
                if ts and ts <= dt_cur: rows.append((ts, d))
            rows.sort(key=lambda x: x[0])
            window = [d for _, d in rows[-self.lookback_points:]]
        else:
            window = self.history[-self.lookback_points:]
        cnt = 0
        for d in reversed(window):
            v = d.get(metric)
            if isinstance(v, (int, float)) and float(v) >= float(threshold):
                cnt += 1
            else:
                break
        return cnt

    @staticmethod
    def _severity(value: float, warn: float, crit: float) -> Optional[str]:
        if value >= crit: return "critical"
        if value >= warn: return "high"
        return None

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metrics: Optional[MetricsData] = state.get("metrics")
        if not metrics:
            state["errors"] = state.get("errors", []) + ["No metrics available for analysis"]
            return state

        ctx: Dict[str, Any] = state.get("context", {}) or {}
        m = metrics.dict()
        anomalies: List[Dict[str, Any]] = []
        # métriques à ignorer pour l'alerting (on ne veut pas d'alertes "uptime élevé")

        EXCLUDED = {"uptime_seconds"}
        # métriques numériques (inclut thread_count)
        for metric_name, value in m.items():
            if metric_name == "service_status": continue
            if metric_name in EXCLUDED: continue
            if not isinstance(value, (int, float)): continue
            th = self.provider.get(metric_name, ctx)
            if not th: continue

            warn = float(th["warning"]); crit = float(th["critical"])
            sev = self._severity(float(value), warn, crit)
            if not sev: continue

            used = crit if sev == "critical" else warn
            consec = self._consecutive_breaches(metric_name, used, m.get("timestamp"))
            if consec < self.consecutive_points: continue

            anomalies.append({
                "metric": metric_name,
                "value": float(value),
                "threshold": used,
                "severity": sev,
                "description": f"{metric_name.replace('_',' ').title()} at {value} ≥ {sev} threshold ({used})",
                "detected_by": th.get("source", "CompositeThresholds"),
                "reason": th.get("reason"),
            })

        # statuts de service
        for svc, status in m.get("service_status", {}).items():
            if status != ServiceStatus.ONLINE:
                sev = "high" if status == ServiceStatus.DEGRADED else "critical"
                anomalies.append({
                    "metric": f"service_{svc}",
                    "value": 0.0,
                    "threshold": 1.0,
                    "severity": sev,
                    "description": f"Service {svc} is {status.value}",
                    "detected_by": "ServiceStatus",
                    "reason": "status != online",
                })

        state["anomalies"] = anomalies
        return state
