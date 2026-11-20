"""HTTP contract of the service (Contrato-IA 2, 4, 7 and 9), offline family."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from insumap_ai.config import Settings
from insumap_ai.main import create_app
from insumap_ai.registry import ModelRegistry
from tests.conftest import (
    PREVIOUS,
    assistant_body,
    falling,
    risk_body,
    series_body,
    settings,
    summary_body,
)


def test_health_and_ready_need_no_token(client: TestClient) -> None:
    assert client.get("/v1/health").json() == {"status": "ok"}
    assert client.get("/v1/ready").json() == {"status": "ready"}


@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic abc", "Bearer "])
def test_rejects_missing_or_invalid_token(
    client: TestClient, header: str | None
) -> None:
    headers = {} if header is None else {"Authorization": header}
    response = client.get("/v1/models", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "INVALID_SERVICE_TOKEN"


def test_accepts_current_and_previous_token(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert client.get("/v1/models", headers=auth).status_code == 200
    previous = {"Authorization": f"Bearer {PREVIOUS}"}
    body = client.get("/v1/models", headers=previous).json()
    assert body["contract_version"] == "1.0.0"
    assert body["provider_family"] == "offline"
    capabilities = {m["capability"] for m in body["models"]}
    assert capabilities == {
        "glucose-forecasts",
        "risk-assessments",
        "clinical-summaries",
        "assistant-replies",
    }
    assert "llm_api_key" not in str(body).lower()


def test_short_token_does_not_start() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        settings(ai_service_token="short")


def test_openai_family_needs_the_three_llm_variables() -> None:
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        settings(ai_provider_family="openai_compatible", llm_model="m")


def test_echoes_request_id_and_generates_one(
    client: TestClient, auth: dict[str, str]
) -> None:
    rid = str(uuid.uuid4())
    response = client.post(
        "/v1/glucose-forecasts",
        json=series_body(falling()),
        headers={**auth, "X-Request-Id": rid},
    )
    assert response.headers["X-Request-Id"] == rid
    assert response.json()["request_id"] == rid
    generated = client.get("/v1/health").headers["X-Request-Id"]
    assert uuid.UUID(generated)


def test_unknown_field_is_a_validation_error(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = series_body(falling())
    body["patient_name"] = "Ana"
    response = client.post("/v1/glucose-forecasts", json=body, headers=auth)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["fields"][0]["path"] == "patient_name"
    assert "Ana" not in response.text


def test_body_too_large(auth: dict[str, str]) -> None:
    ModelRegistry.reset()
    with TestClient(create_app(settings(max_body_bytes=2048))) as small:
        response = small.post(
            "/v1/glucose-forecasts",
            json=series_body(falling(200, 300, 0.0)),
            headers=auth,
        )
    ModelRegistry.reset()
    assert response.status_code == 413
    assert response.json()["error"]["details"] == {"max_bytes": 2048}


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda b: b["series"]["values_mg_dl"].pop(), "LENGTH_MISMATCH"),
        (lambda b: b["series"]["timestamps"].reverse(), "TIMESTAMPS_NOT_INCREASING"),
        (
            lambda b: b["series"]["timestamps"].__setitem__(0, "ayer"),
            "TIMESTAMP_PARSE_ERROR",
        ),
        (
            lambda b: b["series"]["timestamps"].__setitem__(0, "2026-11-14T20:00:00"),
            "TIMESTAMP_PARSE_ERROR",
        ),
        (
            lambda b: b["series"]["values_mg_dl"].__setitem__(3, 900.0),
            "VALUE_OUT_OF_RANGE",
        ),
    ],
)
def test_invalid_series(
    client: TestClient, auth: dict[str, str], mutate, reason
) -> None:  # type: ignore[no-untyped-def]
    body = series_body(falling())
    mutate(body)
    response = client.post("/v1/glucose-forecasts", json=body, headers=auth)
    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "INVALID_SERIES",
        "message": "La serie de glucosa no es válida.",
        "details": {"reason": reason},
    }


def test_too_many_points(auth: dict[str, str]) -> None:
    ModelRegistry.reset()
    with TestClient(create_app(settings(max_forecast_points=40))) as small:
        response = small.post(
            "/v1/glucose-forecasts",
            json=series_body(falling(48, 150, 0.0)),
            headers=auth,
        )
    ModelRegistry.reset()
    assert response.json()["error"]["details"]["reason"] == "TOO_MANY_POINTS"


def test_forecast_answers_the_contract(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = client.post(
        "/v1/glucose-forecasts", json=series_body(falling()), headers=auth
    ).json()
    assert body["is_fallback"] is False and body["fallback_reason"] is None
    assert body["model_name"] == "damped-trend"
    assert len(body["input_hash"]) == 64
    assert body["trend"] == "FALLING"
    assert [p["horizon_minutes"] for p in body["points"]] == [30, 60, 120]
    for point in body["points"]:
        assert point["p10"] <= point["p50"] <= point["p90"]
    hypo = [r["horizon_minutes"] for r in body["risk"]["hypo_probability"]]
    hyper = [r["horizon_minutes"] for r in body["risk"]["hyper_probability"]]
    assert hypo == hyper == [30, 60, 120]
    assert body["explanation"]["data_refs"][0]["kind"] == "READINGS"
    assert "from" in body["explanation"]["data_refs"][0]


def test_same_input_same_output(client: TestClient, auth: dict[str, str]) -> None:
    first = client.post(
        "/v1/glucose-forecasts", json=series_body(falling()), headers=auth
    )
    second = client.post(
        "/v1/glucose-forecasts", json=series_body(falling()), headers=auth
    )
    a, b = first.json(), second.json()
    for key in ("request_id", "generated_at"):
        a.pop(key), b.pop(key)
    assert a == b


def test_short_series_is_insufficient_data(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = client.post(
        "/v1/glucose-forecasts", json=series_body(falling(10)), headers=auth
    ).json()
    assert body["is_fallback"] is True
    assert body["fallback_reason"] == "INSUFFICIENT_DATA"
    assert body["model_version"] == "fallback-linear@1.0.0"
    assert len(body["points"]) == 3


def test_overnight_runs_until_the_end_of_the_night(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = series_body(
        falling(48, 160, 0.0), start="2026-11-14T21:00:00-05:00", mode="OVERNIGHT"
    )
    result = client.post("/v1/glucose-forecasts", json=body, headers=auth).json()
    horizons = [p["horizon_minutes"] for p in result["points"]]
    # Last reading 00:55; the night ends at 06:00.
    assert horizons[-1] == 305
    assert horizons[:5] == [60, 120, 180, 240, 300]
    assert result["trend"] == "STABLE"


def test_risk_answers_the_contract(client: TestClient, auth: dict[str, str]) -> None:
    body = client.post("/v1/risk-assessments", json=risk_body(), headers=auth).json()
    assert body["is_fallback"] is False
    assert body["calibrated"] is False
    assert body["assessment_type"] == "DETERIORATION"
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_tier"] in {"HIGH", "CRITICAL"}
    assert body["explanation"]["factors"]
    assert body["explanation"]["summary"].startswith("Riesgo de deterioro")


def test_summary_offline_uses_the_template(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = client.post(
        "/v1/clinical-summaries", json=summary_body(), headers=auth
    ).json()
    assert body["is_fallback"] is True and body["fallback_reason"] == "AI_ERROR"
    assert body["numeric_check"]["all_matched"] is True
    assert "TIR 48,2 % (meta 70 %)" in body["summary_text"]


def test_assistant_offline_still_escalates(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = client.post(
        "/v1/assistant-replies",
        json=assistant_body("Mi mamá está confundida y no reacciona"),
        headers=auth,
    ).json()
    assert body["escalation"]["level"] == "EMERGENCY"
    assert "EMERGENCY_SYMPTOM" in body["guardrails_triggered"]
    assert body["is_fallback"] is True


def test_message_too_long(
    client: TestClient, auth: dict[str, str], config: Settings
) -> None:
    body = assistant_body("a" * (config.max_message_chars + 1))
    response = client.post("/v1/assistant-replies", json=body, headers=auth)
    assert response.status_code == 422
    assert response.json()["error"]["details"]["fields"][0]["path"] == "message"


def test_not_ready_answers_503(auth: dict[str, str]) -> None:
    ModelRegistry.reset()
    app = create_app(settings())
    client = TestClient(app)  # no lifespan: models never load
    assert client.get("/v1/ready").status_code == 503
    response = client.post("/v1/risk-assessments", json=risk_body(), headers=auth)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"
    ModelRegistry.reset()
