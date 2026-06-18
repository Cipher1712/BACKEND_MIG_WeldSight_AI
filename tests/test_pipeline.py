import math

import numpy as np

from app.features import WINDOW_SIZE, WindowFeatures, extract, windowize
from app.inference import InferencePipeline
from app.quality import compute_quality_index


def healthy_signal() -> list[float]:
    x = np.arange(WINDOW_SIZE) / 750.0
    return (24.0 + 0.35 * np.sin(2 * math.pi * 45 * x)).tolist()


def test_feature_vector_is_finite_and_complete():
    features = extract(healthy_signal())
    assert len(features.to_vector()) == len(WindowFeatures.names()) == 30
    assert np.isfinite(features.to_vector()).all()
    assert features.arc_stability_index > 70
    assert 0 <= features.spectral_entropy <= 1


def test_window_latency_contract():
    windows = list(windowize(healthy_signal() * 2))
    assert len(windows) == 3
    assert WINDOW_SIZE / 750.0 < 0.1


def test_quality_is_monotonic():
    good = compute_quality_index(0.05, 0.05, 0.95, 95, True)["value"]
    bad = compute_quality_index(0.9, 0.9, 0.95, 15, False)["value"]
    assert good > bad


def test_fallback_inference_contract():
    result = InferencePipeline("models-do-not-exist").predict(healthy_signal())
    required = {"quality_index", "severity", "anomaly_score", "physics_label", "ml_label",
                "confidence", "top_features", "recommendation"}
    assert required <= result.keys()
    assert result["model_ready"] is False
