# app.py
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

# IMPORTANT : ces imports supposent que tu as des dossiers "nodes" avec __init__.py dedans
from nodes.ingestion import DataIngestionNode
from nodes.anomaly import AnomalyDetectionNode
from nodes.recommendation import RecommendationGeneratorNode
from nodes.report import ReportGeneratorNode
from thresholds.base import ThresholdProvider   # <-- ici le fix



def create_graph(
    provider: ThresholdProvider,
    history: Optional[List[Dict[str, Any]]] = None,
    consecutive_points: int = 3,
    lookback_points: int = 5,
):
    """
    Construit et compile le graphe LangGraph.
    - provider : ThresholdProvider composite (SLO/percentiles/adaptatif/statique)
    - history  : liste de points (pour l’hystérésis / debounce)
    """
    ingestion = DataIngestionNode()
    anomaly = AnomalyDetectionNode(
        provider=provider,
        history=history or [],
        consecutive_points=consecutive_points,
        lookback_points=lookback_points,
    )
    reco = RecommendationGeneratorNode()   # tu peux injecter un LLM ici si besoin
    report = ReportGeneratorNode()

    g = StateGraph(dict)
    g.add_node("ingest", ingestion.process)
    g.add_node("anomaly", anomaly.process)
    g.add_node("reco", reco.process)
    g.add_node("report", report.process)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "anomaly")
    g.add_edge("anomaly", "reco")
    g.add_edge("reco", "report")
    g.add_edge("report", END)

    return g.compile()
