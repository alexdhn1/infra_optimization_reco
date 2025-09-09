# nodes/recommendation.py
from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional
try:
    from langchain_anthropic import ChatAnthropic
    from langchain.prompts import ChatPromptTemplate
except Exception:
    ChatAnthropic = None
    ChatPromptTemplate = None

class RecommendationGeneratorNode:
    """
    Génère des recommandations à partir des anomalies.
    - Utilise un LLM (Claude Sonnet par défaut) quand dispo.
    - Replie sur des templates par métrique sinon.
    - Classement par priorité (urgent > high > medium > low).
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        if self.llm is None and ChatAnthropic is not None:
            # essaie d'initialiser si la lib est dispo et la clé configurée
            try:
                self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)
            except Exception:
                self.llm = None  # pas grave : on tombera sur les templates

        self.metric_category_map: Dict[str, str] = {
            "cpu_usage": "performance",
            "memory_usage": "performance",
            "latency_ms": "performance",
            "latency_p95": "performance",
            "disk_usage": "reliability",
            "io_wait": "performance",
            "error_rate": "reliability",
            "active_connections": "performance",
            "power_consumption_watts": "cost",
            "network_in_kbps": "performance",
            "network_out_kbps": "performance",
            "temperature_celsius": "reliability",
            "service": "reliability",
        }

        self.templates: Dict[str, Dict[str, Dict[str, Any]]] = {
            "cpu_usage": {
                "high": {"action": "Optimiser et lisser la charge CPU","steps": [
                    "Identifier les top processus CPU (profiling)",
                    "Activer/ajuster l'auto-scaling horizontal",
                    "Ajouter du caching et limiter les recalculs",
                    "Optimiser les requêtes DB les plus coûteuses"]},
                "critical": {"action": "Mitigation immédiate de la saturation CPU","steps": [
                    "Déclencher l'auto-scaling et élargir le pool d'instances",
                    "Shedding de trafic non critique (rate limit/bulkheads)",
                    "Rollback des déploiements récents régressifs",
                    "Augmenter temporairement la taille machine (burstable/spot)"]},
            },
            "memory_usage": {
                "high": {"action": "Réduire la pression mémoire","steps": [
                    "Traquer et corriger les fuites mémoire",
                    "Augmenter les limites ou ajouter du swap contrôlé",
                    "Optimiser structures de données & caches",
                    "Segmenter les workloads gourmands en mémoire"]},
                "critical": {"action": "Prévenir OOM et stabiliser","steps": [
                    "Redémarrer les services fuyards pour récupérer",
                    "Scaler out les composants mémoire-intensifs",
                    "Activer des garde-fous OOM & alertes",
                    "Rollback hotfix si fuite introduite récemment"]},
            },
            "latency_ms": {
                "high": {"action": "Réduire la latence et le temps de réponse","steps": [
                    "Mettre un CDN pour le statique",
                    "Optimiser les index/SQL & ajouter du pooling",
                    "Utiliser des caches (résultats, objets)",
                    "Réplication/edge compute pour proximité utilisateur"]},
                "critical": {"action": "Rétablir la latence sous SLO","steps": [
                    "Contourner les hotspots via routing/feature flags",
                    "Activer read replicas & préchauffer les caches",
                    "Throttler les jobs de fond non critiques",
                    "Rollback du dernier déploiement latence-régressif"]},
            },
            "latency_p95": {
                "high": {"action": "Réduire la latence P95","steps": [
                    "Segmenter endpoints lents & optimiser chemin critique",
                    "Dégrader gracieusement les features non essentielles",
                    "Optimiser sérialisation & taille des payloads",
                    "Activer HTTP keep-alive et compression"]},
                "critical": {"action": "Ramener P95 sous l’objectif","steps": [
                    "Shedding et priorisation des requêtes",
                    "Mettre en lecture seule certaines écritures lourdes",
                    "Activer pré-calculs asynchrones",
                    "Rollback/feature toggle de la feature en cause"]},
            },
            "disk_usage": {
                "high": {"action": "Gérer l’espace disque","steps": [
                    "Rotation et archivage des logs",
                    "Purge des caches/temp inutiles",
                    "Compression & tiering du cold data",
                    "Plan d’extension de capacité"]},
                "critical": {"action": "Éviter l’incident disque plein","steps": [
                    "Purge immédiate du non-essentiel",
                    "Étendre le volume ou attacher du stockage",
                    "Limiter logs verbeux & dumps temporaires",
                    "Ajuster rétentions/backup"]},
            },
            "io_wait": {
                "high": {"action": "Réduire l’attente I/O","steps": [
                    "Identifier hotspots et noisy neighbors",
                    "Tuner pattern I/O (batching, write-back si safe)",
                    "Passer les datasets chauds sur NVMe/SSD",
                    "Limiter jobs batch lourds en heures creuses"]},
                "critical": {"action": "Résoudre la contention stockage","steps": [
                    "Dédier le stockage pour services critiques",
                    "Changer scheduler/queue depth",
                    "Augmenter IOPS/throughput du backend",
                    "Throttler workload I/O non critique"]},
            },
            "error_rate": {
                "high": {"action": "Réduire le taux d’erreurs","steps": [
                    "Inspecter logs/traces & endpoints fautifs",
                    "Retries + backoff + idempotence",
                    "Tests canary & observabilité étendue",
                    "Fix des cas d’erreurs les plus fréquents"]},
                "critical": {"action": "Stopper la vague d’erreurs","steps": [
                    "Circuit breakers / fail-open quand sûr",
                    "Rollback/feature flag immédiat",
                    "Rerouter vers versions saines / canary OK",
                    "Hotfix/patch des exceptions critiques"]},
            },
            "active_connections": {
                "high": {"action": "Gérer la concurrence de connexions","steps": [
                    "Agrandir le pool et tuner les timeouts",
                    "Keep-alive & réutilisation connexions",
                    "Rate limiting sur clients bruyants",
                    "Scaler les frontaux stateless"]},
                "critical": {"action": "Éviter l’épuisement de connexions","steps": [
                    "Shedding sur trafic non critique",
                    "Monter la capacité du LB/ingress",
                    "Sharder services stateful",
                    "Tuner backlog SYN & paramètres kernel"]},
            },
            "power_consumption_watts": {
                "high": {"action": "Optimiser la conso électrique","steps": [
                    "Consolider les workloads sous-utilisés",
                    "Activer profils d’économie CPU",
                    "Planifier batchs en heures creuses",
                    "Auditer services inefficaces"]},
                "critical": {"action": "Réduire la conso critique","steps": [
                    "Arrêter workloads non essentiels",
                    "Répartir la charge sur autres racks/regions",
                    "Vérifier PSU et budgets rack",
                    "Coordonner avec facilities (refroidissement/puissance)"]},
            },
            "network_in_kbps": {
                "high": {"action": "Gérer l’afflux réseau entrant","steps": [
                    "Rate limit/shaping sur sources bruyantes",
                    "Compression quand applicable",
                    "CDN pour offload des assets",
                    "Scaler la couche edge/proxy"]},
                "critical": {"action": "Mitiger saturation entrante","steps": [
                    "Bloquer/mitiger trafic abusif au bord",
                    "Provisionner bande passante/chemins supplémentaires",
                    "Shedding & sampling requêtes",
                    "Activer protections DDoS si besoin"]},
            },
            "network_out_kbps": {
                "high": {"action": "Gérer le trafic sortant","steps": [
                    "Mettre en cache les payloads fréquents",
                    "Réponses delta/partielles",
                    "Batch + compression des réponses",
                    "Auditer endpoints à gros payload"]},
                "critical": {"action": "Mitiger saturation sortante","steps": [
                    "Throttler exports/téléchargements massifs",
                    "Ajouter des points de sortie",
                    "Déporter gros downloads vers objet storage (signed URLs)",
                    "Désactiver temporairement endpoints de debug verbeux"]},
            },
            "temperature_celsius": {
                "high": {"action": "Améliorer la gestion thermique","steps": [
                    "Nettoyer/vérifier refroidissement",
                    "Optimiser circulation d’air",
                    "Réduire temporairement la charge",
                    "Ajouter des solutions de cooling"]},
                "critical": {"action": "Mitiger surchauffe critique","steps": [
                    "Throttle/migrer workloads chauds",
                    "Vérifier ventilateurs et T° ambiante",
                    "Inspecter pannes matérielles",
                    "Déployer du spot cooling / relocaliser"]},
            },
            "service": {
                "high": {"action": "Rétablir le service dégradé","steps": [
                    "Redémarrer de façon contrôlée le composant",
                    "Basculer vers instance/région saine",
                    "Activer circuit breaker côté clients",
                    "Augmenter les ressources temporairement"]},
                "critical": {"action": "Rétablir la disponibilité du service","steps": [
                    "Failover vers réplica/DR immédiatement",
                    "Isoler la cause (réseau/DB/dépendance) et rollback",
                    "Communication statut vers parties prenantes",
                    "Post-mortem et actions correctives"]},
            },
        }

    @staticmethod
    def _safe_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}

    def _category_for(self, metric: str) -> str:
        base = metric.replace("service_", "")
        return self.metric_category_map.get(base, "general")

    def _from_template(self, metric: str, severity: str) -> Dict[str, Any]:
        base = metric.replace("service_", "")
        key = base if base in self.templates else ("service" if metric.startswith("service_") else None)
        if key and severity in self.templates[key]:
            t = self.templates[key][severity]
            return {
                "category": self._category_for(metric),
                "priority": "urgent" if severity == "critical" else "high",
                "action": t["action"],
                "expected_impact": f"Réduire {base.replace('_',' ')} et stabiliser le système",
                "implementation_steps": t["steps"],
            }
        return {
            "category": "general",
            "priority": "medium",
            "action": f"Investigation ciblée sur {metric}",
            "expected_impact": "Amélioration de la performance et de la stabilité",
            "implementation_steps": [
                f"Surveiller {metric} et corréler avec logs/traces",
                "Identifier la cause racine",
                "Déployer la correction ciblée",
            ],
        }

    def _from_llm(self, anomaly: Dict[str, Any], metrics: Optional[Any]) -> Dict[str, Any]:
        if ChatPromptTemplate is None or self.llm is None:
            return self._from_template(anomaly["metric"], "critical" if anomaly.get("severity") == "critical" else "high")

        # ⚠️ Ici on DOUBLE les accolades du JSON d'exemple pour éviter KeyError
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert SRE assistant. "
             "Respond ONLY with a MINIFIED JSON object matching this schema: "
             "{{\"action\":\"...\",\"expected_impact\":\"...\",\"implementation_steps\":[\"...\",\"...\"]}}"),
            ("human", """
Anomaly: {desc}
Metric: {metric}
Current Value: {value}
Severity: {severity}
Context:
- CPU: {cpu}
- MEM: {mem}
- LAT: {lat}
- ERR: {err}
""")
        ])

        cpu = getattr(metrics, "cpu_usage", None) if metrics else None
        mem = getattr(metrics, "memory_usage", None) if metrics else None
        lat = getattr(metrics, "latency_ms", None) if metrics else None
        err = getattr(metrics, "error_rate", None) if metrics else None

        try:
            msgs = prompt.format_messages(
                desc=anomaly.get("description", f"{anomaly['metric']} threshold exceeded"),
                metric=anomaly["metric"],
                value=anomaly.get("value"),
                severity=anomaly.get("severity"),
                cpu=cpu, mem=mem, lat=lat, err=err
            )
        except KeyError:
            # Au cas où, replie sans LLM
            return self._from_template(anomaly["metric"], "critical" if anomaly.get("severity") == "critical" else "high")

        try:
            resp = self.llm.invoke(msgs).content
            data = self._safe_json(str(resp))
            return {
                "category": self._category_for(anomaly["metric"]),
                "priority": "high" if anomaly.get("severity") in ("high", "critical") else "medium",
                "action": data.get("action", "Optimiser l’infrastructure"),
                "expected_impact": data.get("expected_impact", "Performance et fiabilité améliorées"),
                "implementation_steps": data.get("implementation_steps", []),
            }
        except Exception:
            return self._from_template(anomaly["metric"], "critical" if anomaly.get("severity") == "critical" else "high")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        anomalies: List[Dict[str, Any]] = state.get("anomalies", []) or []
        metrics = state.get("metrics")
        recommendations: List[Dict[str, Any]] = []

        if not anomalies:
            state["recommendations"] = []
            return state

        for a in anomalies:
            sev = a.get("severity", "high")
            rec = self._from_llm(a, metrics) if sev in ("high", "critical") else self._from_template(a["metric"], sev)
            recommendations.append(rec)

        order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: order.get(r["priority"], 4))
        state["recommendations"] = recommendations
        return state
