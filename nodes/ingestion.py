# nodes/ingestion.py
from __future__ import annotations
from typing import Dict, Any
from enum import Enum
from pydantic import BaseModel

__all__ = ["ServiceStatus", "MetricsData", "DataIngestionNode"]

class ServiceStatus(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"

class MetricsData(BaseModel):
    timestamp: str
    cpu_usage: float
    memory_usage: float
    latency_ms: int
    disk_usage: float
    network_in_kbps: int
    network_out_kbps: int
    io_wait: float
    thread_count: int
    active_connections: int
    error_rate: float
    uptime_seconds: int
    temperature_celsius: float
    power_consumption_watts: float
    service_status: Dict[str, ServiceStatus]

class DataIngestionNode:
    required_fields = [
        "timestamp",
        "cpu_usage","memory_usage","latency_ms","disk_usage",
        "network_in_kbps","network_out_kbps","io_wait",
        "thread_count","active_connections","error_rate",
        "uptime_seconds","temperature_celsius","power_consumption_watts",
        "service_status",
    ]

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        errors = state.get("errors", [])
        raw = state.get("raw_data") or {}
        missing = [k for k in self.required_fields if k not in raw]
        if missing:
            errors.append(f"Missing required fields: {missing}")
            state["errors"] = errors
            return state
        try:
            state["metrics"] = MetricsData(**raw)
        except Exception as e:
            errors.append(f"Ingestion error: {e}")
            state["errors"] = errors
        return state
