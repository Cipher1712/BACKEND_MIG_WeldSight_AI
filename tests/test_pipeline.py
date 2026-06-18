import numpy as np

from app.features import SAMPLING_RATE_HZ, WINDOW_SIZE, WindowFeatures, extract, windowize
from app.inference import InferencePipeline
from app.quality import compute_quality_index
from app.welding_data import healthy_dataset_paths, iter_real_windows


def real_window() -> list[float]:
    for path in healthy_dataset_paths("data"):
        for window in iter_real_windows(path):
            return window.tolist()
    raise AssertionError("No real valid welding voltage window found in data/")


def test_feature_vector_is_finite_and_complete_on_real_voltage():
    features = extract(real_window())
    assert len(features.to_vector()) == len(WindowFeatures.names()) == 30
    assert np.isfinite(features.to_vector()).all()
    assert 0 <= features.spectral_entropy <= 1
    assert features.median_v >= 8


def test_window_latency_contract_on_real_voltage():
    voltage = real_window() * 2
    windows = list(windowize(voltage))
    assert len(windows) == 3
    assert WINDOW_SIZE / SAMPLING_RATE_HZ < 0.1


def test_quality_is_monotonic():
    good = compute_quality_index(95, 0.95, 0.02, 0.02, 0.02)["value"]
    bad = compute_quality_index(20, 0.05, 0.95, 0.95, 0.95)["value"]
    assert good > bad


def test_fallback_inference_contract_on_real_voltage():
    result = InferencePipeline("models-do-not-exist").predict(real_window())
    required = {"quality_score", "anomaly_score", "status", "diagnosis", "top_contributors"}
    assert required <= result.keys()
    assert result["model_ready"] is False
