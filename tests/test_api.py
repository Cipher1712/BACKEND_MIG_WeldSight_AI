import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app import recordings
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
        expected = {
            "quality_score", "anomaly_score", "status", "diagnosis", "top_contributors",
            "confidence", "arc_instability_score", "spatter_risk_score",
            "burn_through_risk_score", "low_heat_input_score", "quality_breakdown",
        }
        assert expected <= frame.keys()
        assert frame["prediction"] in {
            "Healthy Arc", "Arc Instability", "Spatter Risk",
            "Burn Through Risk", "Low Heat Input Risk",
        }
        assert frame["physics_label"] == frame["prediction"] == frame["ml_label"] == frame["status"]
        assert 0.0 <= frame["confidence"] <= 1.0
        assert len(frame["top_contributors"]) == 3
        assert all(isinstance(item, str) for item in frame["top_contributors"])


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
                assert "arc_instability_score" in frame


def test_http_telemetry_ingestion_and_polling_contract():
    samples = real_samples(64)
    with TestClient(app) as client:
        response = client.post("/telemetry", json={
            "voltage": samples,
            "distance_mm": 7.25,
            "arc_on": True,
            "timestamp": 123456,
        })
        assert response.status_code == 200
        assert response.json()["frames_processed"] >= 1

        latest = client.get("/telemetry/latest")
        assert latest.status_code == 200
        assert latest.json()["voltage"] == samples
        assert latest.json()["timestamp"] == 123456
        assert latest.json()["latest_inference"]["event_timestamp_ms"] < 123456
        assert latest.json()["latest_inference"]["distance_source"] == "estimated"
        assert latest.json()["latest_inference"]["position_label"] == "Relative Position"

        metrics = client.get("/metrics/latest")
        assert metrics.status_code == 200
        assert {"quality_index", "stability", "anomalies"} <= metrics.json().keys()

        events = client.get("/events/latest")
        assert events.status_code == 200
        assert isinstance(events.json(), list)


def test_http_telemetry_rejects_malformed_payload():
    with TestClient(app) as client:
        response = client.post("/telemetry", json={
            "voltage": [],
            "distance_mm": 0,
            "arc_on": True,
            "timestamp": 123456,
        })
        assert response.status_code == 422
        assert "detail" in response.json()


def test_recording_session_generates_raw_csv_and_metadata():
    with TestClient(app) as client:
        client.post("/recording/stop")
        started = client.post("/recording/start", json={"notes": "pytest recording"})
        assert started.status_code == 200
        session_id = started.json()["session_id"]

        response = client.post("/telemetry", json={
            "voltage": [14.0, 14.5, 15.0, 15.5],
            "encoder_counts": 10,
            "encoder_mm_per_count": 0.25,
            "arc_on": True,
            "timestamp": 1003,
            "timestamps_ms": [1000, 1001, 1002, 1003],
        })
        assert response.status_code == 200

        stopped = client.post("/recording/stop")
        assert stopped.status_code == 200
        metadata = stopped.json()
        assert metadata["session_id"] == session_id
        assert metadata["sample_count"] == 4
        assert metadata["duration_ms"] == 3
        assert metadata["distance_mm"] == 2.5
        assert metadata["distance_source"] == "Encoder"
        assert metadata["csv_size_bytes"] > 0

        listed = client.get("/recordings")
        assert listed.status_code == 200
        assert any(row["session_id"] == session_id for row in listed.json())

        downloaded = client.get(f"/recordings/{session_id}/download")
        assert downloaded.status_code == 200
        lines = downloaded.text.strip().splitlines()
        assert lines[0] == "timestamp,voltage,encoder_count,distance_mm,sample_index"
        assert lines[1].startswith("1000,14.0,10.0,2.5,0")
        assert len(lines) == 5

        unmarked_training = client.post(f"/training/from-recording/{session_id}")
        assert unmarked_training.status_code == 400
        assert "healthy_baseline" in unmarked_training.json()["detail"]

        marked = client.post(
            f"/recordings/{session_id}/healthy-baseline",
            json={"healthy_baseline": True, "notes": "healthy pytest recording"},
        )
        assert marked.status_code == 200
        assert marked.json()["healthy_baseline"] is True
        marked_without_body = client.post(f"/recordings/{session_id}/healthy-baseline")
        assert marked_without_body.status_code == 200
        assert marked_without_body.json()["healthy_baseline"] is True

        deleted = client.delete(f"/recordings/{session_id}")
        assert deleted.status_code == 200


def test_training_from_healthy_recording_writes_reports_and_marks_trained(monkeypatch):
    def fake_train(matrix, output_dir, **_):
        assert matrix.shape[0] >= 1
        return {
            "training_windows": int(matrix.shape[0]),
            "validation_windows": 1,
            "validation_reconstruction": 0.01,
            "artifacts": ["vae.pt", "scaler.pkl", "isolation_forest.pkl", "anomaly_threshold.json"],
        }

    model_dir = Path("models") / "_pytest_reports"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    monkeypatch.setattr(recordings.recording_manager, "model_dir", model_dir)
    monkeypatch.setattr(recordings, "train_healthy_models", fake_train)
    monkeypatch.setattr(recordings.RecordingManager, "_deploy_staged_artifacts", lambda self, staging_dir: None)
    monkeypatch.setattr(recordings.RecordingManager, "_validate_staged_model", staticmethod(
        lambda staging_dir, features, matrix: {
            "passed": True,
            "failure_reason": None,
            "false_positive_rate": 0.0,
            "detection_rate": 1.0,
            "threshold_separation": 0.5,
            "auc": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "healthy_window_count": 2,
            "training_window_count": int(matrix.shape[0]),
            "validation_window_count": 2,
        }
    ))

    try:
        with TestClient(app) as client:
            client.post("/recording/stop")
            started = client.post("/recording/start", json={"healthy_baseline": True})
            assert started.status_code == 200
            session_id = started.json()["session_id"]
            samples = [15.0 + (index % 5) * 0.1 for index in range(96)]
            response = client.post("/telemetry", json={
                "voltage": samples,
                "encoder_counts": list(range(96)),
                "encoder_mm_per_count": 0.5,
                "arc_on": True,
                "timestamp": 2095,
                "timestamps_ms": list(range(2000, 2096)),
            })
            assert response.status_code == 200
            assert client.post("/recording/stop").status_code == 200

            trained = client.post(f"/training/from-recording/{session_id}")
            assert trained.status_code == 200
            body = trained.json()
            assert body["status"] == "success"
            assert body["model_reloaded"] is True
            assert body["reports"] == [
                "evaluation_report.json",
                "pipeline_audit.json",
                "reliability_report.json",
                "training_report.json",
                "validation_report.json",
            ]
            assert (model_dir / "reports" / "training_report.json").exists()
            status = client.get("/training/status")
            assert status.status_code == 200
            assert status.json()["status"] == "Completed"
            recording = client.get(f"/recordings/{session_id}").json()
            assert recording["trained"] is True
            assert recording["model_version_used"]
            inference = client.post("/api/infer", json={
                "material": "mild_steel",
                "thickness_mm": 6,
                "voltage": real_samples(96),
            })
            assert inference.status_code == 200
            assert inference.json()["frames"][0]["prediction"]
            assert client.delete(f"/recordings/{session_id}").status_code == 200
    finally:
        if model_dir.exists():
            shutil.rmtree(model_dir)
