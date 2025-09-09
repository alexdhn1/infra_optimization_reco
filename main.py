# main.py
import asyncio, json, yaml
from datetime import datetime
from typing import Dict, Any, List
from app import create_graph

# Providers
from thresholds.static_provider import StaticThresholdProvider
from thresholds.slo_provider import SLOBurnRateProvider
from thresholds.percentile_provider import PercentileThresholdProvider
from thresholds.adaptive_provider import AdaptiveStatsProvider
from thresholds.composite import CompositeThresholdProvider

def load_data(fname: str = "rapport.json") -> List[Dict[str, Any]]:
    try:
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # fallback minimal
        return [{
            "timestamp":"2025-09-08T10:00:00Z",
            "cpu_usage":75, "memory_usage":70, "latency_ms":220, "error_rate":0.012,
            "thread_count": 950,
            "service_status":{"api":"online"}
        }]

def load_cfg(path: str = "config/thresholds.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

async def run_once(history: List[Dict[str, Any]], cfg: Dict[str, Any]):
    # Composite provider (ordre = SLO -> Percentiles -> Adaptatif -> Statique)
    provider = CompositeThresholdProvider([
        SLOBurnRateProvider(cfg.get("slos", {}), history),
        PercentileThresholdProvider(history, p_warn=0.90, p_crit=0.99),
        AdaptiveStatsProvider(history, alpha=0.25),
        StaticThresholdProvider(cfg.get("static", {})),
    ])

    # Hystérésis depuis la config (défauts: 3/5)
    hyst = cfg.get("hysteresis", {})
    consecutive = int(hyst.get("consecutive_points", 3))
    lookback = int(hyst.get("lookback_points", 5))

    app = create_graph(
        provider=provider,
        history=history,
        consecutive_points=consecutive,
        lookback_points=lookback,
    )

    # on prend le DERNIER point comme “courant”
    current = history[-1]
    state = {"raw_data": current, "errors": [], "context": {}}  # tu peux mettre {"service": "api_gateway"} ici

    result = await app.ainvoke(state)
    report = result["report"]

    # Affiche + sauvegarde
    print(json.dumps(report, indent=2))
    with open("infrastructure_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "total_data_points": len(history),
            "report": report,
        }, f, indent=2)

if __name__ == "__main__":
    data = load_data("rapport.json")
    cfg = load_cfg("config/thresholds.yaml")
    asyncio.run(run_once(data, cfg))
