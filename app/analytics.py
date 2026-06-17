"""PCA + DBSCAN cluster projection."""
from typing import List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def project_and_cluster(feature_rows: List[List[float]], eps: float = 0.7, min_samples: int = 4):
    if not feature_rows or len(feature_rows) < 6:
        return {"embeddings": [], "labels": [], "clusters": 0, "noise": 0}
    X = np.asarray(feature_rows, dtype=float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    emb = PCA(n_components=2).fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emb)
    return {
        "embeddings": emb.tolist(), "labels": labels.tolist(),
        "clusters": int(len({int(l) for l in labels if l >= 0})),
        "noise": int((labels == -1).sum()),
    }