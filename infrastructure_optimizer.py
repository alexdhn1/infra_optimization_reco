"""
Infrastructure Optimization System for Technical Infrastructure
A modular application using LangGraph for anomaly detection and recommendation generation
(Revision with fixes & improvements: service status handling, required fields, broader templates,
robust LLM JSON parsing, safer time-series analysis, corrected types and minor refactors)
"""

import json
import re
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated, Tuple
from datetime import datetime
from enum import Enum
import statistics

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ============= Data Models =============

class ServiceStatus(str, Enum):
    """Service health status enumeration"""
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class MetricsData(BaseModel):
    """Infrastructure metrics data model"""
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
    # Cast incoming strings like "online" to Enum automatically
    service_status: Dict[str, ServiceStatus]


class Anomaly(BaseModel):
    """Detected anomaly model"""
    metric: str
    current_value: float
    threshold: float
    severity: str  # low, medium, high, critical
    description: str


class Recommendation(BaseModel):
    """Infrastructure optimization recommendation"""
    category: str  # performance, reliability, cost, security, general
    priority: str  # low, medium, high, urgent
    action: str
    expected_impact: str
    implementation_steps: List[str]


# ============= State Definition =============

class InfrastructureState(TypedDict):
    """State object that flows through the graph"""
    raw_data: Dict[str, Any]
    metrics: MetricsData
    anomalies: List[Anomaly]
    recommendations: List[Recommendation]
    report: Dict[str, Any]
    errors: List[str]


# ============= Node Functions =============

class DataIngestionNode:
    """Node 1: Data ingestion and validation"""

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
        """Ingest and validate infrastructure data"""
        errors = []

        try:
            raw_data = state.get('raw_data', {}) or {}

            # Validate required fields
            missing_fields = [f for f in self.required_fields if f not in raw_data]
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")
                state['errors'] = errors
                return state

            # Create metrics object (Pydantic will coerce service_status values to Enum)
            metrics = MetricsData(**raw_data)
            state['metrics'] = metrics

            print(f"✅ Data ingested successfully at {metrics.timestamp}")

        except Exception as e:
            errors.append(f"Ingestion error: {str(e)}")
            state['errors'] = errors

        return state


class AnomalyDetectionNode:
    """Node 2: Detect anomalies in infrastructure metrics"""

    def __init__(self):
        # Define thresholds for anomaly detection
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 85},
            'memory_usage': {'warning': 75, 'critical': 90},
            'latency_ms': {'warning': 200, 'critical': 500},
            'disk_usage': {'warning': 80, 'critical': 90},
            'io_wait': {'warning': 10, 'critical': 20},
            'error_rate': {'warning': 0.01, 'critical': 0.05},
            'temperature_celsius': {'warning': 70, 'critical': 85},
            'power_consumption_watts': {'warning': 300, 'critical': 400},
            'active_connections': {'warning': 100, 'critical': 200},
            'network_in_kbps': {'warning': 80000, 'critical': 120000},   # example thresholds
            'network_out_kbps': {'warning': 80000, 'critical': 120000},  # adjust to your env
        }

    def _determine_severity(self, value: float, warning: float, critical: float, is_inverted: bool = False) -> str:
        """Determine anomaly severity based on thresholds"""
        if is_inverted:  # For metrics where lower is worse (unused currently)
            if value <= critical:
                return "critical"
            elif value <= warning:
                return "high"
            else:
                return "low"
        else:  # For metrics where higher is worse
            if value >= critical:
                return "critical"
            elif value >= warning:
                return "high"
            else:
                return "low"

    def process(self, state: InfrastructureState) -> InfrastructureState:
        """Detect anomalies in the metrics"""
        anomalies: List[Anomaly] = []
        metrics = state.get('metrics')

        if not metrics:
            state['errors'] = state.get('errors', []) + ["No metrics available for analysis"]
            return state

        metrics_dict = metrics.dict()

        # Check each metric against thresholds
        for metric_name, thresholds in self.thresholds.items():
            if metric_name in metrics_dict:
                value = metrics_dict[metric_name]
                warning_threshold = thresholds['warning']
                critical_threshold = thresholds['critical']

                # Check if metric exceeds thresholds
                if value >= warning_threshold:
                    severity = self._determine_severity(value, warning_threshold, critical_threshold)

                    anomaly = Anomaly(
                        metric=metric_name,
                        current_value=float(value),
                        threshold=float(critical_threshold if severity == "critical" else warning_threshold),
                        severity=severity,
                        description=f"{metric_name.replace('_', ' ').title()} is at {value}, exceeding {severity} threshold"
                    )
                    anomalies.append(anomaly)

        # Check service status (Enums now)
        service_status: Dict[str, ServiceStatus] = metrics_dict.get('service_status', {})  # type: ignore
        for service, status in service_status.items():
            if status != ServiceStatus.ONLINE:
                anomaly = Anomaly(
                    metric=f"service_{service}",
                    current_value=0,
                    threshold=1,
                    severity="high" if status == ServiceStatus.DEGRADED else "critical",
                    description=f"Service {service} is {status.value}"
                )
                anomalies.append(anomaly)

        state['anomalies'] = anomalies

        print(f"🔍 Detected {len(anomalies)} anomalies")
        for anomaly in anomalies[:10]:
            print(f"  - {anomaly.severity.upper()}: {anomaly.description}")

        return state


class RecommendationGeneratorNode:
    """Node 3: Generate optimization recommendations based on anomalies"""

    def __init__(self, llm=None):
        # Use Claude or OpenAI for intelligent recommendations
        self.llm = llm or ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)

        # Predefined recommendation templates (expanded coverage)
        self.recommendation_templates: Dict[str, Dict[str, Dict[str, Any]]] = {
            'cpu_usage': {
                'high': {
                    'action': 'Scale horizontally or optimize CPU-intensive processes',
                    'steps': [
                        'Identify top CPU-consuming processes',
                        'Consider horizontal scaling with load balancer',
                        'Optimize application code for CPU efficiency',
                        'Implement caching strategies'
                    ]
                },
                'critical': {
                    'action': 'Immediate CPU optimization required',
                    'steps': [
                        'Enable auto-scaling immediately',
                        'Distribute load across multiple instances',
                        'Review and optimize database queries',
                        'Consider upgrading to higher CPU tier'
                    ]
                }
            },
            'memory_usage': {
                'high': {
                    'action': 'Optimize memory allocation and usage',
                    'steps': [
                        'Analyze memory leaks in applications',
                        'Increase memory limits or add swap space',
                        'Implement memory caching strategies',
                        'Review and optimize data structures'
                    ]
                },
                'critical': {
                    'action': 'Mitigate memory saturation urgently',
                    'steps': [
                        'Restart leaking services to reclaim memory',
                        'Temporarily scale out memory-heavy components',
                        'Enable memory pressure alerts and OOM prevention',
                        'Hotfix known leak patterns or rollback recent release'
                    ]
                }
            },
            'latency_ms': {
                'high': {
                    'action': 'Reduce network latency and response times',
                    'steps': [
                        'Implement CDN for static content',
                        'Optimize database queries and indexes',
                        'Enable connection pooling',
                        'Consider geographic distribution'
                    ]
                },
                'critical': {
                    'action': 'Urgent latency remediation',
                    'steps': [
                        'Bypass hotspots via feature flags or routing rules',
                        'Activate read replicas / cache hot shards',
                        'Throttle non-critical background jobs',
                        'Roll back recent latency-regressing deployments'
                    ]
                }
            },
            'disk_usage': {
                'high': {
                    'action': 'Manage disk space and implement cleanup policies',
                    'steps': [
                        'Archive old logs and data',
                        'Implement log rotation policies',
                        'Clean temporary files and caches',
                        'Consider expanding storage capacity'
                    ]
                },
                'critical': {
                    'action': 'Prevent imminent disk full events',
                    'steps': [
                        'Purge non-essential data immediately',
                        'Extend volume size or attach additional storage',
                        'Enable compression and tiering for cold data',
                        'Audit backup/retention policies'
                    ]
                }
            },
            'temperature_celsius': {
                'high': {
                    'action': 'Improve cooling and thermal management',
                    'steps': [
                        'Check and clean cooling systems',
                        'Optimize airflow in server room',
                        'Reduce computational load temporarily',
                        'Consider additional cooling solutions'
                    ]
                },
                'critical': {
                    'action': 'Mitigate critical overheating',
                    'steps': [
                        'Throttle workloads or migrate hot services',
                        'Verify fan speeds, heat sinks, and ambient temperature',
                        'Inspect for hardware faults',
                        'Deploy spot cooling or relocate hardware'
                    ]
                }
            },
            'error_rate': {
                'high': {
                    'action': 'Reduce application error rate',
                    'steps': [
                        'Inspect error logs and SLO dashboards',
                        'Identify top failing endpoints/queries',
                        'Implement retries with backoff and idempotency',
                        'Add canary tests and expand observability (traces/metrics)'
                    ]
                },
                'critical': {
                    'action': 'Mitigate critical error spike',
                    'steps': [
                        'Enable circuit breakers / fail-open where safe',
                        'Roll back recent deployments or feature flags',
                        'Shift traffic to healthy instances or previous version',
                        'Hotfix known exceptions and raise alerting thresholds temporarily'
                    ]
                }
            },
            'active_connections': {
                'high': {
                    'action': 'Manage connection concurrency',
                    'steps': [
                        'Increase connection pool size and tune timeouts',
                        'Enable keep-alive and connection reuse',
                        'Add rate limiting on bursty clients',
                        'Scale out stateless frontends'
                    ]
                },
                'critical': {
                    'action': 'Prevent connection exhaustion',
                    'steps': [
                        'Drop or throttle non-critical traffic',
                        'Scale horizontally and add load balancer capacity',
                        'Shard stateful services to spread load',
                        'Review SYN backlog and kernel networking params'
                    ]
                }
            },
            'power_consumption_watts': {
                'high': {
                    'action': 'Optimize power usage',
                    'steps': [
                        'Consolidate low-utilization workloads',
                        'Enable CPU frequency scaling / power profiles',
                        'Schedule batch jobs during off-peak energy windows',
                        'Audit inefficient services and dependencies'
                    ]
                },
                'critical': {
                    'action': 'Address critical power draw',
                    'steps': [
                        'Reduce non-essential workloads immediately',
                        'Distribute compute to secondary racks/regions',
                        'Verify PSU health and rack power budgets',
                        'Engage facilities for cooling/power review'
                    ]
                }
            },
            'network_in_kbps': {
                'high': {
                    'action': 'Handle high inbound bandwidth',
                    'steps': [
                        'Introduce ingress rate limiting for noisy sources',
                        'Enable compression where applicable',
                        'Offload static assets to CDN',
                        'Scale edge/proxy layer'
                    ]
                },
                'critical': {
                    'action': 'Mitigate inbound saturation',
                    'steps': [
                        'Block/shape abusive traffic at edge',
                        'Provision additional bandwidth or routes',
                        'Enable request sampling and shedding',
                        'Activate DDoS protections if applicable'
                    ]
                }
            },
            'network_out_kbps': {
                'high': {
                    'action': 'Handle high outbound bandwidth',
                    'steps': [
                        'Cache frequently requested payloads',
                        'Enable delta/partial responses',
                        'Batch and compress responses',
                        'Audit large payload endpoints'
                    ]
                },
                'critical': {
                    'action': 'Mitigate egress saturation',
                    'steps': [
                        'Throttle bulk data transfers / exports',
                        'Route traffic via additional egress points',
                        'Move large downloads to object storage with pre-signed URLs',
                        'Temporarily disable verbose debug endpoints'
                    ]
                }
            },
            'io_wait': {
                'high': {
                    'action': 'Reduce IO wait',
                    'steps': [
                        'Identify disk hotspots and noisy neighbors',
                        'Tune filesystem and database IO patterns',
                        'Enable write-back caching where safe',
                        'Move hot data to faster storage tiers (NVMe/SSD)'
                    ]
                },
                'critical': {
                    'action': 'Resolve storage contention urgently',
                    'steps': [
                        'Throttle IO-heavy background jobs',
                        'Migrate critical services to dedicated storage',
                        'Change IO scheduler / queue depth',
                        'Scale storage bandwidth or IOPS limits'
                    ]
                }
            }
        }

        # Category hints per metric
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
        """Try to parse JSON from LLM output robustly."""
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

    def _generate_smart_recommendation(self, anomaly: Anomaly, metrics: MetricsData) -> Recommendation:
        """Generate intelligent recommendations using LLM"""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert SRE/infra assistant. "
             "Respond ONLY with a MINIFIED JSON object matching the schema: "
             '{"action": "...", "expected_impact": "...", "implementation_steps": ["...", "..."]}. '
             "No code fences, no extra text."),
            ("human", """
Based on the following infrastructure anomaly, generate a specific optimization recommendation:

Anomaly: {anomaly_desc}
Metric: {metric}
Current Value: {current_value}
Severity: {severity}

Current Infrastructure Status:
- CPU Usage: {cpu}%
- Memory Usage: {memory}%
- Latency: {latency}ms
- Error Rate: {error_rate}

Provide a concrete action plan with expected impact.
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
            response = self.llm.invoke(messages)
            content = getattr(response, "content", response)  # AIMessage or raw str
            if isinstance(content, list):
                content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            content = str(content)
            recommendation_data = self._parse_json_safe(content)

            return Recommendation(
                category=self.metric_category_map.get(anomaly.metric.replace('service_', ''), "general"),
                priority="high" if anomaly.severity in ["critical", "high"] else "medium",
                action=recommendation_data.get('action', 'Optimize infrastructure'),
                expected_impact=recommendation_data.get('expected_impact', 'Improved performance and reliability'),
                implementation_steps=recommendation_data.get('implementation_steps', [])
            )
        except Exception:
            # Fallback to template-based recommendations
            return self._get_template_recommendation(anomaly)

    def _get_template_recommendation(self, anomaly: Anomaly) -> Recommendation:
        """Get recommendation from predefined templates"""
        # Normalize metric: drop "service_" prefix used for service anomalies
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

        # Default recommendation
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

    def process(self, state: InfrastructureState) -> InfrastructureState:
        """Generate recommendations based on detected anomalies"""
        anomalies = state.get('anomalies', [])
        metrics = state.get('metrics')
        recommendations: List[Recommendation] = []

        if not anomalies:
            print("✨ No anomalies detected - system running optimally!")
            state['recommendations'] = []
            return state

        # Generate recommendations for each anomaly
        for anomaly in anomalies:
            if anomaly.severity in ['critical', 'high']:
                # Use smart recommendation for critical issues if metrics available
                rec = self._generate_smart_recommendation(anomaly, metrics) if metrics else self._get_template_recommendation(anomaly)
            else:
                # Use template for lower severity
                rec = self._get_template_recommendation(anomaly)

            recommendations.append(rec)

        # Sort by priority
        priority_order = {'urgent': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))

        state['recommendations'] = recommendations

        print(f"💡 Generated {len(recommendations)} recommendations")
        for rec in recommendations[:3]:  # Show top 3
            print(f"  - [{rec.priority.upper()}] {rec.action}")

        return state


class ReportGeneratorNode:
    """Node 4: Generate final optimization report"""

    def process(self, state: InfrastructureState) -> InfrastructureState:
        """Generate comprehensive optimization report"""
        metrics = state.get('metrics')
        anomalies = state.get('anomalies', [])
        recommendations = state.get('recommendations', [])

        # Calculate system health score
        health_score = self._calculate_health_score(metrics, anomalies)

        # Group recommendations by category
        rec_by_category: Dict[str, List[Dict[str, Any]]]= {}
        for rec in recommendations:
            rec_by_category.setdefault(rec.category, []).append({
                'priority': rec.priority,
                'action': rec.action,
                'expected_impact': rec.expected_impact,
                'implementation_steps': rec.implementation_steps
            })

        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'score': health_score,
                'status': self._get_health_status(health_score),
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
            'recommendations': rec_by_category,
            'immediate_actions': [
                rec.action for rec in recommendations
                if rec.priority in ['urgent', 'high']
            ][:5]  # Top 5 immediate actions
        }

        state['report'] = report

        # Print summary
        print("\n" + "="*60)
        print("📊 INFRASTRUCTURE OPTIMIZATION REPORT")
        print("="*60)
        print(f"System Health Score: {health_score}/100 ({report['system_health']['status']})")
        print(f"Critical Issues: {report['system_health']['critical_issues']}")
        print(f"Total Anomalies: {report['system_health']['total_anomalies']}")
        print(f"\nImmediate Actions Required:")
        for i, action in enumerate(report['immediate_actions'], 1):
            print(f"  {i}. {action}")
        print("="*60)

        return state

    def _calculate_health_score(self, metrics: MetricsData, anomalies: List[Anomaly]) -> int:
        """Calculate system health score (0-100)"""
        if not metrics:
            return 0

        base_score = 100

        # Deduct points based on anomalies
        for anomaly in anomalies:
            if anomaly.severity == 'critical':
                base_score -= 20
            elif anomaly.severity == 'high':
                base_score -= 10
            else:
                base_score -= 5

        # Additional deductions for critical metrics
        if metrics.cpu_usage > 90:
            base_score -= 10
        if metrics.memory_usage > 90:
            base_score -= 10
        if metrics.error_rate > 0.05:
            base_score -= 15

        return max(0, min(100, base_score))

    def _get_health_status(self, score: int) -> str:
        """Get health status based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Critical"


# ============= Graph Construction =============

def create_infrastructure_graph() -> StateGraph:
    """Create the LangGraph workflow for infrastructure optimization"""

    # Initialize nodes
    ingestion_node = DataIngestionNode()
    anomaly_node = AnomalyDetectionNode()
    recommendation_node = RecommendationGeneratorNode()
    report_node = ReportGeneratorNode()

    # Create graph
    workflow = StateGraph(InfrastructureState)

    # Add nodes
    workflow.add_node("ingest_data", ingestion_node.process)
    workflow.add_node("detect_anomalies", anomaly_node.process)
    workflow.add_node("generate_recommendations", recommendation_node.process)
    workflow.add_node("generate_report", report_node.process)

    # Define edges (workflow)
    workflow.set_entry_point("ingest_data")
    workflow.add_edge("ingest_data", "detect_anomalies")
    workflow.add_edge("detect_anomalies", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# ============= Main Application =============

async def process_infrastructure_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to process infrastructure data"""

    # Create the graph
    app = create_infrastructure_graph()

    # Initial state
    initial_state: InfrastructureState = {
        'raw_data': data,
        'metrics': None,  # type: ignore
        'anomalies': [],
        'recommendations': [],
        'report': {},
        'errors': []
    }

    # Run the graph
    result = await app.ainvoke(initial_state)

    return result['report']


def load_data_from_file(filename: str = "rapport.json") -> List[Dict[str, Any]]:
    """Load infrastructure data from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} data points from {filename}")
        return data
    except FileNotFoundError:
        print(f"⚠️ File {filename} not found, using sample data")
        return load_sample_data()
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return []


def load_sample_data() -> List[Dict[str, Any]]:
    """Load sample infrastructure data for testing"""
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
            }
        }
    ]


def analyze_time_series_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns in the time series data (safe against missing keys)"""
    if not data:
        return {}

    def series(key: str) -> List[float]:
        vals = []
        for d in data:
            v = d.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    cpu_series = series('cpu_usage')
    memory_series = series('memory_usage')
    latency_series = series('latency_ms')
    error_series = series('error_rate')

    def stats(arr: List[float]) -> Dict[str, Any]:
        if not arr:
            return {'mean': 0, 'max': 0, 'min': 0, 'stdev': 0}
        return {
            'mean': statistics.mean(arr),
            'max': max(arr),
            'min': min(arr),
            'stdev': statistics.stdev(arr) if len(arr) > 1 else 0,
        }

    service_disruptions = {
        'database_offline': sum(1 for d in data if d.get('service_status', {}).get('database') == 'offline'),
        'api_degraded': sum(1 for d in data if d.get('service_status', {}).get('api_gateway') == 'degraded'),
        'cache_degraded': sum(1 for d in data if d.get('service_status', {}).get('cache') == 'degraded')
    }

    patterns = {
        'cpu': {
            **stats(cpu_series),
            'critical_peaks': sum(1 for x in cpu_series if x > 90)
        },
        'memory': {
            **stats(memory_series),
            'critical_peaks': sum(1 for x in memory_series if x > 85)
        },
        'latency': {
            **stats(latency_series),
            'high_latency_periods': sum(1 for x in latency_series if x > 300)
        },
        'errors': {
            **stats(error_series),
            'total_high_error_periods': sum(1 for x in error_series if x > 0.05)
        },
        'service_disruptions': service_disruptions
    }

    return patterns


async def process_batch_data(data_batch: List[Dict[str, Any]], batch_size: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process data in batches for efficiency"""
    reports: List[Dict[str, Any]] = []
    critical_incidents: List[Dict[str, Any]] = []

    for i in range(0, len(data_batch), batch_size):
        batch = data_batch[i:i+batch_size]

        for data_point in batch:
            report = await process_infrastructure_data(data_point)
            reports.append(report)

            # Track critical incidents
            if report['system_health']['critical_issues'] > 0:
                critical_incidents.append({
                    'timestamp': data_point.get('timestamp'),
                    'issues': report['system_health']['critical_issues'],
                    'health_score': report['system_health']['score']
                })

    return reports, critical_incidents


async def main():
    """Main execution function"""
    print("🚀 Starting Infrastructure Optimization System")
    print("=" * 60)

    # Load data from file
    all_data = load_data_from_file("rapport.json")

    if not all_data:
        print("❌ No data to process")
        return

    # Analyze overall patterns
    print("\n📊 Analyzing time series patterns...")
    patterns = analyze_time_series_patterns(all_data)

    print("\n📈 Infrastructure Statistics:")
    cpu_mean = patterns.get('cpu', {}).get('mean', 0.0)
    mem_mean = patterns.get('memory', {}).get('mean', 0.0)
    lat_mean = patterns.get('latency', {}).get('mean', 0.0)
    cpu_peaks = patterns.get('cpu', {}).get('critical_peaks', 0)
    mem_peaks = patterns.get('memory', {}).get('critical_peaks', 0)
    high_lat_periods = patterns.get('latency', {}).get('high_latency_periods', 0)
    svc = patterns.get('service_disruptions', {})
    print(f"  CPU Usage: avg={cpu_mean:.1f}%, peaks>{90}%: {cpu_peaks}")
    print(f"  Memory Usage: avg={mem_mean:.1f}%, peaks>{85}%: {mem_peaks}")
    print(f"  Latency: avg={lat_mean:.0f}ms, high periods: {high_lat_periods}")
    print(f"  Service Disruptions: DB offline={svc.get('database_offline', 0)}, API degraded={svc.get('api_degraded', 0)}")

    # Process critical incidents only (CPU > 90% or Error Rate > 0.1)
    critical_data = [d for d in all_data if (isinstance(d.get('cpu_usage'), (int, float)) and d['cpu_usage'] > 90) or (isinstance(d.get('error_rate'), (int, float)) and d['error_rate'] > 0.1)]

    print(f"\n🚨 Found {len(critical_data)} critical incidents to analyze")
    print("-" * 60)

    all_reports: List[Dict[str, Any]] = []

    # Process up to first 5 critical incidents in parallel
    to_process = critical_data[:5]
    if to_process:
        print(f"⚙️  Processing {len(to_process)} critical incidents in parallel...")
        tasks = [process_infrastructure_data(d) for d in to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for data, report in zip(to_process, results):
            if isinstance(report, Exception):
                print(f"❌ Error processing incident at {data.get('timestamp')}: {report}")
                continue

            all_reports.append(report)

            # Save individual critical report
            timestamp_str = str(data.get('timestamp', datetime.now().isoformat())).replace(':', '-').replace('T', '_').split('.')[0]
            filename = f"critical_report_{timestamp_str}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)

    # Create summary report
    summary_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_data_points': len(all_data),
        'critical_incidents_analyzed': len(critical_data),
        'time_period': {
            'start': all_data[0].get('timestamp'),
            'end': all_data[-1].get('timestamp')
        },
        'infrastructure_statistics': patterns,
        'critical_incidents': [
            {
                'timestamp': d.get('timestamp'),
                'cpu': d.get('cpu_usage'),
                'memory': d.get('memory_usage'),
                'error_rate': d.get('error_rate'),
                'services_affected': [k for k, v in (d.get('service_status', {}) or {}).items() if v != 'online']
            }
            for d in critical_data
        ],
        'recommendations_summary': {
            'immediate_actions': [
                "Implement auto-scaling for CPU peaks above 90%",
                "Optimize memory allocation during high-traffic periods",
                "Improve API gateway resilience to prevent degradation",
                "Set up predictive monitoring for database offline events",
                "Implement circuit breakers for high error rate scenarios"
            ],
            'long_term_improvements': [
                "Upgrade infrastructure capacity based on peak usage patterns",
                "Implement distributed caching to reduce latency",
                "Review and optimize database connection pooling",
                "Deploy redundant API gateway instances",
                "Implement comprehensive observability stack"
            ]
        }
    }

    # Save summary report
    with open('infrastructure_analysis_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✅ Processed {len(all_data)} data points")
    print(f"🚨 Identified {len(critical_data)} critical incidents")
    print(f"📁 Summary report saved to: infrastructure_analysis_summary.json")
    print(f"📁 Individual critical reports saved (if any)")
    print("=" * 60)


if __name__ == "__main__":
    """
    Architecture Overview:
    ---------------------
    This application uses LangGraph to create a modular, multi-node processing pipeline:

    1. Data Ingestion Node: Validates and structures incoming metrics
    2. Anomaly Detection Node: Identifies performance and reliability issues
    3. Recommendation Generator Node: Creates actionable optimization strategies
    4. Report Generator Node: Compiles comprehensive analysis report

    Technology Choices:
    ------------------
    - LangGraph: For orchestrating the multi-step workflow with state management
    - LangChain: For LLM integration and intelligent recommendation generation
    - Pydantic: For robust data validation and type safety
    - Async/Await: For handling real-time data streams efficiently

    Key Features:
    ------------
    - Threshold-based anomaly detection with severity classification
    - Intelligent recommendations using LLM or template-based fallback
    - Comprehensive health scoring system
    - JSON-formatted reports for easy integration
    - Real-time processing capability
    """

    # Run the application
    asyncio.run(main())
