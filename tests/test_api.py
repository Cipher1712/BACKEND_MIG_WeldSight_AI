from fastapi.testclient import TestClient

from app.main import app
from app.welding_data import healthy_dataset_paths, iter_real_windows


def real_samples(count: int = 96) -> list[float]:
    values: list[float] = []
    for path in healthy_dataset_paths("data"):
        for window in iter_real_windows(path):
            values.extend(window.tolist())
            if len(values) >= count:
                return values[:count]
    raise AssertionError("No real valid welding voltage samples found in data/")


def test_health_and_batch_contract_with_real_voltage():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert {"model_ready", "vae_loaded", "scaler_loaded"} <= health.json().keys()
        response = client.post("/api/infer", json={
            "material": "mild_steel", "thickness_mm": 6, "voltage": real_samples(),
        })
        assert response.status_code == 200
        frame = response.json()["frames"][0]
        expected = {"quality_score", "anomaly_score", "status", "diagnosis", "top_contributors"}
        assert expected <= frame.keys()


def test_live_websocket_contract_with_real_voltage():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as live:
            with client.websocket_connect("/ws/stream") as stream:
                stream.send_json({"material": "mild_steel", "thickness_mm": 6})
                stream.send_json({"voltage": real_samples(64), "distance_mm": 12.5, "arc_on": True})
                frame = live.receive_json()
                assert frame["distance_mm"] == 12.5
                assert "quality_score" in frame
                assert "diagnosis" in frame
