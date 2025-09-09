# app.py (révision intégrant ThresholdProvider, hysteresis, SLOs, percentiles et YAML)
import json
import re
import asyncio
from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from thresholds import (
    ThresholdProvider,
    load_threshold_stack,
    HysteresisConfig,
    PercentileThresholdProvider,
)

# ============= Data Models =============

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

class Anomaly(BaseModel):
    metric: str
    current_value: float
    threshold: float
    severity: str
    description: str

class Recommendation(BaseModel):
    category: str
    priority: str
    action: str
    expected_impact: str
    implementation_steps: List[str]

class InfrastructureState(TypedDict):
    raw_data: Dict[str, Any]
    metrics: Optional[MetricsData]
    anomalies: List[Anomaly]
    recommendations: List[Recommendation]
    report: Dict[str, Any]
    errors: List[str]
    context: Dict[str, Any]  # e.g., {"service": "api_gateway"}

# ============= Node Functions =============

class DataIngestionNode:
    def __init__(self):
        self.required_fields = [
            "timestamp",
            'cpu_usage', 'memory_usage', 'latency_ms', 'disk_usage',
            'network_in_kbps', 'network_out_kbps', 'io_wait',
            'thread_count', 'active_connections', 'error_rate',
            'uptime_seconds', 'temperature_celsius', 'power_consumption_watts',
            "service_status",
        ]

    def process(self, state: InfrastructureState) -> InfrastructureState:
        errors = []
        try:
            raw_data = state.get('raw_data', {}) or {}
            missing = [f for f in self.required_fields if f not in raw_data]
            if missing:
                errors.append(f"Missing required fields: {missing}")
                state['errors'] = errors
                return state
            metrics = MetricsData(**raw_data)
            state['metrics'] = metrics
            print(f"✅ Data ingested @ {metrics.timestamp}")
        except Exception as e:
            errors.append(f"Ingestion error: {str(e)}")
            state['errors'] = errors
        return state

class AnomalyDetectionNode:
    def __init__(self, provider: ThresholdProvider, hysteresis: HysteresisConfig, history: List[Dict[str, Any]]):
        self.provider = provider
        self.hysteresis = hysteresis
        # keep a reference to percentile provider if we need consecutive breach check with history
        self.pct_provider = None
        if isinstance(provider, PercentileThresholdProvider):
            self.pct_provider = provider
        # or attempt to find nested PercentileThresholdProvider
        elif hasattr(provider, "providers"):
            for p in getattr(provider, "providers", []):
                if isinstance(p, PercentileThresholdProvider):
                    self.pct_provider = p
                    break
        self.metric_category_map = {
            'cpu_usage': "performance",
            'memory_usage': "performance",
            'latency_ms': "performance",
            'disk_usage': "reliability",
            'temperature_celsius': "reliability",
            'error_rate': "reliability",
            'active_connections': "performance",
            'power_consumption_watts': "cost",
            'network_in_kbps': "performance",
            'network_out_kbps': "performance",
            'io_wait': "performance",
        }
        self.history = history or []

    def _severity(self, value: float, warning: float, critical: float) -> str:
        if value >= critical:
            return "critical"
        if value >= warning:
            return "high"
        return "low"

    def _breach_confirmed(self, metric: str, threshold: float, ts: Optional[str], service: Optional[str]) -> bool:
        """Apply hysteresis: require N consecutive breaches within lookback window (incl. current point if present in history)."""
        if self.hysteresis.consecutive_points <= 1:
            return True
        if not self.pct_provider:
            # No history available; cannot confirm hysteresis, be conservative and accept breach
            return True
        count = self.pct_provider.consecutive_breaches(
            metric=metric,
            threshold=threshold,
            current_ts=ts,
            service=service,
            lookback_points=self.hysteresis.lookback_points,
        )
        return count >= self.hysteresis.consecutive_points

    def process(self, state: InfrastructureState) -> InfrastructureState:
        anomalies: List[Anomaly] = []
        metrics = state.get('metrics')
        if not metrics:
            state['errors'] = state.get('errors', []) + ["No metrics available for analysis"]
            return state

        # context can carry service name, env, etc.
        context = state.get("context", {})
        service_ctx = context.get("service")  # optional

        m = metrics.dict()

        # numeric metrics (skip service_status here)
        for metric_name, value in m.items():
            if metric_name == "service_status":
                continue
            if not isinstance(value, (int, float)):
                continue

            th = self.provider.get(metric_name, {"service": service_ctx})
            if not th:
                continue

            # Apply hysteresis for warning and critical checks
            sev = None
            if value >= th["critical"]:
                if self._breach_confirmed(metric_name, th["critical"], metrics.timestamp, service_ctx):
                    sev = "critical"
            elif value >= th["warning"]:
                if self._breach_confirmed(metric_name, th["warning"], metrics.timestamp, service_ctx):
                    sev = "high"

            if sev:
                anomalies.append(Anomaly(
                    metric=metric_name,
                    current_value=float(value),
                    threshold=float(th["critical"] if sev == "critical" else th["warning"]),
                    severity=sev,
                    description=f"{metric_name.replace('_',' ').title()} at {value} >= {sev} threshold"
                ))

        # service statuses
        for svc, status in m.get('service_status', {}).items():
            if status != ServiceStatus.ONLINE:
                sev = "high" if status == ServiceStatus.DEGRADED else "critical"
                anomalies.append(Anomaly(
                    metric=f"service_{svc}",
                    current_value=0,
                    threshold=1,
                    severity=sev,
                    description=f"Service {svc} is {status.value}"
                ))

        state['anomalies'] = anomalies
        print(f"🔍 Detected {len(anomalies)} anomalies")
        for a in anomalies[:10]:
            print(f"  - {a.severity.upper()}: {a.description}")
        return state

class RecommendationGeneratorNode:
    def __init__(self, llm=None):
        self.llm = llm or ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.3)
        self.recommendation_templates: Dict[str, Dict[str, Dict[str, Any]]] = {
            # (same expanded templates as précédemment)
            'cpu_usage': {
                'high': {'action': 'Scale horizontally or optimize CPU-intensive processes','steps': ['Identify top CPU-consuming processes','Consider horizontal scaling with load balancer','Optimize application code for CPU efficiency','Implement caching strategies']},
                'critical': {'action': 'Immediate CPU optimization required','steps': ['Enable auto-scaling immediately','Distribute load across multiple instances','Review and optimize database queries','Consider upgrading to higher CPU tier']}
            },
            'memory_usage': {
                'high': {'action': 'Optimize memory allocation and usage','steps': ['Analyze memory leaks in applications','Increase memory limits or add swap space','Implement memory caching strategies','Review and optimize data structures']},
                'critical': {'action': 'Mitigate memory saturation urgently','steps': ['Restart leaking services to reclaim memory','Temporarily scale out memory-heavy components','Enable memory pressure alerts and OOM prevention','Hotfix known leak patterns or rollback recent release']}
            },
            'latency_ms': {
                'high': {'action': 'Reduce network latency and response times','steps': ['Implement CDN for static content','Optimize database queries and indexes','Enable connection pooling','Consider geographic distribution']},
                'critical': {'action': 'Urgent latency remediation','steps': ['Bypass hotspots via feature flags or routing rules','Activate read replicas / cache hot shards','Throttle non-critical background jobs','Roll back recent latency-regressing deployments']}
            },
            'disk_usage': {
                'high': {'action': 'Manage disk space and implement cleanup policies','steps': ['Archive old logs and data','Implement log rotation policies','Clean temporary files and caches','Consider expanding storage capacity']},
                'critical': {'action': 'Prevent imminent disk full events','steps': ['Purge non-essential data immediately','Extend volume size or attach additional storage','Enable compression and tiering for cold data','Audit backup/retention policies']}
            },
            'temperature_celsius': {
                'high': {'action': 'Improve cooling and thermal management','steps': ['Check and clean cooling systems','Optimize airflow in server room','Reduce computational load temporarily','Consider additional cooling solutions']},
                'critical': {'action': 'Mitigate critical overheating','steps': ['Throttle workloads or migrate hot services','Verify fan speeds, heat sinks, and ambient temperature','Inspect for hardware faults','Deploy spot cooling or relocate hardware']}
            },
            'error_rate': {
                'high': {'action': 'Reduce application error rate','steps': ['Inspect error logs and SLO dashboards','Identify top failing endpoints/queries','Implement retries with backoff and idempotency','Add canary tests and expand observability (traces/metrics)']},
                'critical': {'action': 'Mitigate critical error spike','steps': ['Enable circuit breakers / fail-open where safe','Roll back recent deployments or feature flags','Shift traffic to healthy instances or previous version','Hotfix known exceptions and raise alerting thresholds temporarily']}
            },
            'active_connections': {
                'high': {'action': 'Manage connection concurrency','steps': ['Increase connection pool size and tune timeouts','Enable keep-alive and connection reuse','Add rate limiting on bursty clients','Scale out stateless frontends']},
                'critical': {'action': 'Prevent connection exhaustion','steps': ['Drop or throttle non-critical traffic','Scale horizontally and add load balancer capacity','Shard stateful services to spread load','Review SYN backlog and kernel networking params']}
            },
            'power_consumption_watts': {
                'high': {'action': 'Optimize power usage','steps': ['Consolidate low-utilization workloads','Enable CPU frequency scaling / power profiles','Schedule batch jobs during off-peak energy windows','Audit inefficient services and dependencies']},
                'critical': {'action': 'Address critical power draw','steps': ['Reduce non-essential workloads immediately','Distribute compute to secondary racks/regions','Verify PSU health and rack power budgets','Engage facilities for cooling/power review']}
            },
            'network_in_kbps': {
                'high': {'action': 'Handle high inbound bandwidth','steps': ['Introduce ingress rate limiting for noisy sources','Enable compression where applicable','Offload static assets to CDN','Scale edge/proxy layer']},
                'critical': {'action': 'Mitigate inbound saturation','steps': ['Block/shape abusive traffic at edge','Provision additional bandwidth or routes','Enable request sampling and shedding','Activate DDoS protections if applicable']}
            },
            'network_out_kbps': {
                'high': {'action': 'Handle high outbound bandwidth','steps': ['Cache frequently requested payloads','Enable delta/partial responses','Batch and compress responses','Audit large payload endpoints']},
                'critical': {'action': 'Mitigate egress saturation','steps': ['Throttle bulk data transfers / exports','Route traffic via additional egress points','Move large downloads to object storage with pre-signed URLs','Temporarily disable verbose debug endpoints']}
            },
            'io_wait': {
                'high': {'action': 'Reduce IO wait','steps': ['Identify disk hotspots and noisy neighbors','Tune filesystem and database IO patterns','Enable write-back caching where safe','Move hot data to faster storage tiers (NVMe/SSD)']},
                'critical': {'action': 'Resolve storage contention urgently','steps': ['Throttle IO-heavy background jobs','Migrate critical services to dedicated storage','Change IO scheduler / queue depth','Scale storage bandwidth or IOPS limits']}
            }
        }
        self.metric_category_map = {
            'cpu_usage': "performance",
            'memory_usage': "performance",
            'latency_ms': "performance",
            'disk_usage': "reliability",
            'temperature_celsius': "reliability",
            'error_rate': "reliability",
            'active_connections': "performance",
            'power_consumption_watts': "cost",
            'network_in_kbps': "performance",
            'network_out_kbps': "performance",
            'io_wait': "performance",
        }

    def _parse_json_safe(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}

    def _get_template_recommendation(self, anomaly: Anomaly) -> Recommendation:
        metric_base = anomaly.metric.replace('service_', '')
        templates = self.recommendation_templates.get(metric_base)
        severity_key = 'critical' if anomaly.severity == 'critical' else 'high'
        if templates and severity_key in templates:
            template = templates[severity_key]
            return Recommendation(
                category=self.metric_category_map.get(metric_base, "general"),
                priority="urgent" if anomaly.severity == "critical" else "high",
                action=template['action'],
                expected_impact=f"Reduce {metric_base.replace('_', ' ')} and improve system stability",
                implementation_steps=template['steps']
            )
        return Recommendation(
            category="general",
            priority="medium",
            action=f"Investigate and optimize {anomaly.metric}",
            expected_impact="Improved system performance",
            implementation_steps=[
                f"Monitor {anomaly.metric} closely",
                "Identify root cause",
                "Implement targeted optimization"
            ]
        )

    def _generate_smart_recommendation(self, anomaly: Anomaly, metrics: MetricsData) -> Recommendation:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert SRE/infra assistant. "
             "Respond ONLY with a MINIFIED JSON object matching the schema: "
             '{"action": "...", "expected_impact": "...", "implementation_steps": ["...", "..."]}.'),
            ("human", """
Anomaly: {anomaly_desc}
Metric: {metric}
Current Value: {current_value}
Severity: {severity}

Current:
- CPU: {cpu}%
- Mem: {memory}%
- Latency: {latency}ms
- ErrorRate: {error_rate}
""")
        ])
        try:
            messages = prompt.format_messages(
                anomaly_desc=anomaly.description,
                metric=anomaly.metric,
                current_value=anomaly.current_value,
                severity=anomaly.severity,
                cpu=metrics.cpu_usage,
                memory=metrics.memory_usage,
                latency=metrics.latency_ms,
                error_rate=metrics.error_rate
            )
            response = (self.llm.invoke(messages)).content
            data = self._parse_json_safe(str(response))
            return Recommendation(
                category=self.metric_category_map.get(anomaly.metric.replace('service_', ''), "general"),
                priority="high" if anomaly.severity in ["critical", "high"] else "medium",
                action=data.get('action', 'Optimize infrastructure'),
                expected_impact=data.get('expected_impact', 'Improved performance and reliability'),
                implementation_steps=data.get('implementation_steps', [])
            )
        except Exception:
            return self._get_template_recommendation(anomaly)

    def process(self, state: InfrastructureState) -> InfrastructureState:
        anomalies = state.get('anomalies', [])
        metrics = state.get('metrics')
        recs: List[Recommendation] = []
        if not anomalies:
            print("✨ No anomalies detected - system running optimally!")
            state['recommendations'] = []
            return state
        for a in anomalies:
            if a.severity in ("critical", "high") and metrics:
                recs.append(self._generate_smart_recommendation(a, metrics))
            else:
                recs.append(self._get_template_recommendation(a))
        priority_order = {'urgent': 0, 'high': 1, 'medium': 2, 'low': 3}
        recs.sort(key=lambda x: priority_order.get(x.priority, 4))
        state['recommendations'] = recs
        print(f"💡 Generated {len(recs)} recommendations")
        for r in recs[:3]:
            print(f"  - [{r.priority.upper()}] {r.action}")
        return state

class ReportGeneratorNode:
    def process(self, state: InfrastructureState) -> InfrastructureState:
        metrics = state.get('metrics')
        anomalies = state.get('anomalies', [])
        recommendations = state.get('recommendations', [])
        score = self._health_score(metrics, anomalies)
        rec_by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for rec in recommendations:
            rec_by_cat.setdefault(rec.category, []).append({
                'priority': rec.priority,
                'action': rec.action,
                'expected_impact': rec.expected_impact,
                'implementation_steps': rec.implementation_steps
            })
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'score': score,
                'status': self._status(score),
                'critical_issues': len([a for a in anomalies if a.severity == 'critical']),
                'high_priority_issues': len([a for a in anomalies if a.severity == 'high']),
                'total_anomalies': len(anomalies)
            },
            'metrics_summary': {
                'cpu_usage': metrics.cpu_usage if metrics else 0,
                'memory_usage': metrics.memory_usage if metrics else 0,
                'latency_ms': metrics.latency_ms if metrics else 0,
                'error_rate': metrics.error_rate if metrics else 0,
                'uptime_hours': (metrics.uptime_seconds / 3600) if metrics else 0
            },
            'anomalies_detected': [
                {
                    'metric': a.metric,
                    'severity': a.severity,
                    'current_value': a.current_value,
                    'threshold': a.threshold,
                    'description': a.description
                } for a in anomalies
            ],
            'recommendations': rec_by_cat,
            'immediate_actions': [
                rec.action for rec in recommendations
                if rec.priority in ['urgent', 'high']
            ][:5]
        }
        state['report'] = report
        print("\n" + "="*60)
        print("📊 INFRASTRUCTURE OPTIMIZATION REPORT")
        print("="*60)
        print(f"System Health Score: {score}/100 ({report['system_health']['status']})")
        print(f"Critical Issues: {report['system_health']['critical_issues']}")
        print(f"Total Anomalies: {report['system_health']['total_anomalies']}")
        print("\nImmediate Actions:")
        for i, a in enumerate(report['immediate_actions'], 1):
            print(f"  {i}. {a}")
        print("="*60)
        return state

    def _health_score(self, metrics: Optional[MetricsData], anomalies: List[Anomaly]) -> int:
        if not metrics:
            return 0
        score = 100
        for a in anomalies:
            if a.severity == 'critical':
                score -= 20
            elif a.severity == 'high':
                score -= 10
            else:
                score -= 5
        if metrics.cpu_usage > 90:
            score -= 10
        if metrics.memory_usage > 90:
            score -= 10
        if metrics.error_rate > 0.05:
            score -= 15
        return max(0, min(100, score))

    def _status(self, s: int) -> str:
        if s >= 90: return "Excellent"
        if s >= 75: return "Good"
        if s >= 60: return "Fair"
        if s >= 40: return "Poor"
        return "Critical"

# ============= Graph Construction =============

def create_infrastructure_graph(
    provider: ThresholdProvider,
    hysteresis: HysteresisConfig,
    history: List[Dict[str, Any]],
) -> StateGraph:
    ingestion_node = DataIngestionNode()
    anomaly_node = AnomalyDetectionNode(provider, hysteresis, history)
    recommendation_node = RecommendationGeneratorNode()
    report_node = ReportGeneratorNode()

    workflow = StateGraph(InfrastructureState)
    workflow.add_node("ingest_data", ingestion_node.process)
    workflow.add_node("detect_anomalies", anomaly_node.process)
    workflow.add_node("generate_recommendations", recommendation_node.process)
    workflow.add_node("generate_report", report_node.process)
    workflow.set_entry_point("ingest_data")
    workflow.add_edge("ingest_data", "detect_anomalies")
    workflow.add_edge("detect_anomalies", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()

# ============= Main Application =============

def load_data_from_file(filename: str = "rapport.json") -> List[Dict[str, Any]]:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return load_sample_data()
    except json.JSONDecodeError:
        return []

def load_sample_data() -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": "2023-10-01T12:00:00Z",
            "cpu_usage": 85,
            "memory_usage": 70,
            "latency_ms": 250,
            "disk_usage": 65,
            "network_in_kbps": 1200,
            "network_out_kbps": 900,
            "io_wait": 5,
            "thread_count": 150,
            "active_connections": 45,
            "error_rate": 0.02,
            "uptime_seconds": 360000,
            "temperature_celsius": 65,
            "power_consumption_watts": 250,
            "service_status": {
                "database": "online",
                "api_gateway": "degraded",
                "cache": "online"
            },
            # optional: tag a record with service context for percentile grouping
            "service": "api_gateway"
        }
    ]

async def process_infrastructure_data(data: Dict[str, Any], provider: ThresholdProvider, hysteresis: HysteresisConfig, history: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    app = create_infrastructure_graph(provider, hysteresis, history)
    initial_state: InfrastructureState = {
        'raw_data': data,
        'metrics': None,
        'anomalies': [],
        'recommendations': [],
        'report': {},
        'errors': [],
        'context': context or {}
    }
    result = await app.ainvoke(initial_state)
    return result['report']

async def main():
    print("🚀 Starting Infrastructure Optimization System (adaptive thresholds + hysteresis)")
    all_data = load_data_from_file("rapport.json")
    # Build providers stack from YAML + history
    provider, hcfg = load_threshold_stack("thresholds.yaml", all_data)

    # Optional: choose a global service context; or derive from each data point
    default_context = {"service": "api_gateway"}

    # Process all points (demo)
    reports: List[Dict[str, Any]] = []
    for d in all_data:
        ctx = {"service": d.get("service")} if d.get("service") else default_context
        rep = await process_infrastructure_data(d, provider, hcfg, all_data, ctx)
        reports.append(rep)

    with open('infrastructure_analysis_summary.json', 'w') as f:
        json.dump({
            "analysis_timestamp": datetime.now().isoformat(),
            "total_data_points": len(all_data),
            "reports": reports
        }, f, indent=2)

    print("✅ Saved summary: infrastructure_analysis_summary.json")

if __name__ == "__main__":
    asyncio.run(main())
