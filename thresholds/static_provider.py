import sys
import os

# Ajouter le répertoire racine du projet au sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from typing import Dict, Any, Optional
from thresholds.base import ThresholdProvider

class StaticThresholdProvider(ThresholdProvider):
    def __init__(self, table: Dict[str, Dict[str, float]]):
        self.table = table

    def get(self, metric: str, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        return self.table.get(metric)
