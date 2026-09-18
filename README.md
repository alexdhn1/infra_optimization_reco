# Infrastructure Optimization Reco

Pipeline d'analyse d'infrastructure qui détecte les anomalies sur des métriques système (CPU, mémoire, latence, disque, réseau, erreurs…) et génère des recommandations de remédiation priorisées, avec un score de santé global.

## Fonctionnement

Le pipeline est construit avec **LangGraph** en 4 étapes séquentielles :

```
ingestion → détection d'anomalies → génération de recommandations → rapport
```

1. **Ingestion** (`nodes/ingestion.py`) — valide et structure les métriques brutes (Pydantic) à partir d'un snapshot JSON.
2. **Détection d'anomalies** (`nodes/anomaly.py`) — compare chaque métrique à un seuil calculé dynamiquement (voir ci-dessous) et applique une hystérésis anti-flapping.
3. **Recommandations** (`nodes/recommendation.py`) — pour chaque anomalie `high`/`critical`, interroge un LLM (Claude) pour une recommandation contextualisée ; repli automatique sur des templates prédéfinis par métrique si le LLM échoue ou n'est pas disponible.
4. **Rapport** (`nodes/report.py`) — calcule un score de santé (0-100), regroupe les recommandations par catégorie (performance / reliability / cost) et liste les actions immédiates prioritaires.

### Seuils dynamiques (`thresholds/`)

Plutôt que des seuils statiques uniques, les seuils sont calculés par plusieurs providers combinables :

- **Statique** (`static_provider.py`) — table de seuils warning/critical définie en YAML, avec overrides par service.
- **SLO** (`slo_provider.py`) — dérive les seuils d'un objectif de SLO (ex. latence P95 à 200ms → warning à 1.1×, critique à 1.5×).
- **Percentiles** (`percentile_provider.py`) — seuils calculés sur l'historique (P90/P99) par métrique et par service.
- **Adaptatif** (`adaptive_provider.py`) — statistiques glissantes (moyenne/écart-type) qui s'ajustent au comportement récent du système.
- **Composite** (`composite.py`) — combine tous les providers actifs et retient le seuil le plus strict.

Une **hystérésis** (nombre de breaches consécutifs requis dans une fenêtre de lookback, configurable) évite de déclencher une alerte sur un pic isolé.

## Structure

```
infra_optimization_reco/
├── main.py                    # point d'entrée : charge les données + la config, lance le pipeline
├── app.py                     # construction et compilation du graphe LangGraph
├── nodes/
│   ├── ingestion.py            # validation des métriques (Pydantic)
│   ├── anomaly.py              # détection + hystérésis
│   ├── recommendation.py       # génération LLM + fallback templates
│   └── report.py               # score de santé + rapport structuré
├── thresholds/
│   ├── base.py                  # interface ThresholdProvider
│   ├── static_provider.py
│   ├── slo_provider.py
│   ├── percentile_provider.py
│   ├── adaptive_provider.py
│   └── composite.py
├── config/
│   └── thresholds.yaml         # seuils statiques, objectifs SLO, config d'hystérésis
└── test.py
```

## Utilisation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."   # optionnel : sans clé, repli sur les templates de recommandation

python main.py
```

Le script lit un historique de métriques (`rapport.json`, ou données d'exemple si absent), applique la pile de seuils définie dans `config/thresholds.yaml`, et produit `infrastructure_analysis_summary.json` contenant le rapport complet (score de santé, anomalies détectées, recommandations par catégorie, actions immédiates).

## Choix techniques

- **LangGraph** pour un pipeline explicite, traçable et facile à étendre (ajouter un nœud sans toucher au reste).
- **Composite de providers de seuils** plutôt qu'un seuil fixe : combine connaissance métier (SLO, statique) et signal statistique (percentiles, adaptatif), pour rester pertinent quand le trafic ou la charge évoluent.
- **Hystérésis** pour réduire le bruit d'alerting (pas d'alerte sur un seul point hors seuil).
- **LLM avec repli déterministe** : les recommandations critiques bénéficient d'un contexte généré par Claude, mais le système reste fonctionnel sans clé API ou en cas d'échec du LLM (aucune dépendance dure sur le LLM pour produire un rapport).
