import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict

class MultivariateDetector:
    def __init__(self, history: List[Dict[str, float]]):
        self.metrics = [k for k in history[0] if isinstance(history[0][k], (int,float))]
        X = np.array([[d[m] for m in self.metrics] for d in history])
        self.model = IsolationForest(contamination=0.01).fit(X)

    def score(self, point: Dict[str,float]) -> float:
        X = np.array([[point[m] for m in self.metrics]])
        return -self.model.score_samples(X)[0]  # plus grand = plus anormal
