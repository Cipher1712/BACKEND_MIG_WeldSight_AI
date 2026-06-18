import math

from fastapi.testclient import TestClient

from app.main import app


def samples(count: int = 96) -> list[float]:
    return [24.0 + 0.3 * math.sin(2 * math.pi * 45 * index / 750) for index in range(count)]


def test_health_and_batch_contract():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["window_size"] == 64
        response = client.post("/api/infer", json={
            "material": "mild_steel", "thickness_mm": 6, "voltage": samples(),
        })
        assert response.status_code == 200
        frame = response.json()["frames"][0]
        expected = {"timestamp", "voltage", "quality_index", "severity", "anomaly_score",
                    "physics_label", "ml_label", "confidence", "top_features", "recommendation"}
        assert expected <= frame.keys()


def test_live_websocket_contract():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as live:
            with client.websocket_connect("/ws/stream") as stream:
                stream.send_json({"material": "mild_steel", "thickness_mm": 6})
                stream.send_json({"voltage": samples(64), "distance_mm": 12.5, "arc_on": True})
                frame = live.receive_json()
                assert frame["distance_mm"] == 12.5
                assert "quality_index" in frame
                assert "ml_label" in frame
