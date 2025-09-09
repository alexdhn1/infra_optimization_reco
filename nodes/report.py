# nodes/report.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime

SEVERITY_WEIGHT = {"critical": 20, "high": 10, "medium": 5, "low": 3}

class ReportGeneratorNode:
    """
    Construit un rapport complet :
    - score de santé
    - résumé des métriques numériques présentes
    - anomalies (avec source si dispo)
    - recommandations groupées par catégorie
    - top actions immédiates
    """

    # ———— helpers ————

    def _health_score(self, metrics: Optional[Any], anomalies: List[Dict[str, Any]]) -> int:
        base = 100
        for a in anomalies:
            base -= SEVERITY_WEIGHT.get(a.get("severity", "high"), 8)
        # déductions supplémentaires (si métriques présentes)
        if getattr(metrics, "cpu_usage", 0) and metrics.cpu_usage > 90:
            base -= 10
        if getattr(metrics, "memory_usage", 0) and metrics.memory_usage > 90:
            base -= 10
        if getattr(metrics, "error_rate", 0) and metrics.error_rate > 0.05:
            base -= 15
        return max(0, min(100, base))

    def _status(self, score: int) -> str:
        if score >= 90: return "Excellent"
        if score >= 75: return "Good"
        if score >= 60: return "Fair"
        if score >= 40: return "Poor"
        return "Critical"

    def _numeric_summary(self, metrics_obj: Optional[Any]) -> Dict[str, float]:
        """
        Résume toutes les métriques numériques disponibles sur l'objet Pydantic (MetricsData).
        On ne hard-code pas la liste pour éviter d'en louper.
        """
        out: Dict[str, float] = {}
        if not metrics_obj:
            return out
        data = metrics_obj.dict()
        for k, v in data.items():
            if isinstance(v, (int, float)):
                out[k] = float(v)
            # uptime_seconds → uptime_hours
            if k == "uptime_seconds" and isinstance(v, (int, float)):
                out["uptime_hours"] = round(float(v) / 3600.0, 2)
        return out

    # ———— entrée du nœud ————

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        anomalies: List[Dict[str, Any]] = state.get("anomalies", []) or []
        recommendations: List[Dict[str, Any]] = state.get("recommendations", []) or []
        metrics = state.get("metrics")

        score = self._health_score(metrics, anomalies)

        # groupement des recos par catégorie
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for r in recommendations:
            by_cat.setdefault(r["category"], []).append({
                "priority": r["priority"],
                "action": r["action"],
                "expected_impact": r["expected_impact"],
                "implementation_steps": r.get("implementation_steps", []),
            })

        # actions immédiates (top 5)
        immediate = [r["action"] for r in recommendations if r["priority"] in ("urgent", "high")][:5]

        # enrichir anomalies avec 'detected_by' si présent (traçabilité)
        anomalies_out = []
        for a in anomalies:
            anomalies_out.append({
                "metric": a.get("metric"),
                "severity": a.get("severity"),
                "current_value": a.get("value", a.get("current_value")),
                "threshold": a.get("threshold"),
                "description": a.get("description"),
                "detected_by": a.get("detected_by"),  # optionnel
            })

        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system_health": {
                "score": score,
                "status": self._status(score),
                "critical_issues": sum(1 for a in anomalies if a.get("severity") == "critical"),
                "high_priority_issues": sum(1 for a in anomalies if a.get("severity") == "high"),
                "total_anomalies": len(anomalies),
            },
            "metrics_summary": self._numeric_summary(metrics),
            "anomalies_detected": anomalies_out,
            "recommendations": by_cat,
            "immediate_actions": immediate,
        }

        state["report"] = report
        return state
