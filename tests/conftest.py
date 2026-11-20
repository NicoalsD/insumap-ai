"""Shared fixtures: offline settings, a scripted LLM and contract examples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from insumap_ai.config import Settings
from insumap_ai.main import create_app
from insumap_ai.providers.base import (
    ModelInfo,
    Prompt,
    TextGenerationError,
    TextGenerator,
)
from insumap_ai.registry import ModelRegistry

TOKEN = "t" * 40
PREVIOUS = "p" * 40
SUBJECT = "sub_5f2c8e7a9b1d4c3e8a6f0d2b7c9e1a43"


class ScriptedTextGenerator(TextGenerator):
    """Answers with the queued replies and records every prompt it received."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.prompts: list[Prompt] = []

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(name="scripted-llm", version="0.1.0+prompt-v1")

    def generate(self, prompt: Prompt, *, json_output: bool) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise TextGenerationError("EMPTY_SCRIPT")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "ai_service_token": SecretStr(TOKEN),
        "ai_service_token_previous": SecretStr(PREVIOUS),
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def config() -> Settings:
    return settings()


@pytest.fixture
def client(config: Settings) -> Iterator[TestClient]:
    ModelRegistry.reset()
    with TestClient(create_app(config)) as test_client:
        yield test_client
    ModelRegistry.reset()


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def series_body(
    values: list[float],
    *,
    start: str = "2026-11-14T20:00:00-05:00",
    mode: str = "SHORT_TERM",
) -> dict[str, Any]:
    first = datetime.fromisoformat(start)
    stamps = [
        (first + timedelta(minutes=5 * i)).isoformat() for i in range(len(values))
    ]
    body: dict[str, Any] = {
        "subject_ref": SUBJECT,
        "mode": mode,
        "series": {"timestamps": stamps, "values_mg_dl": values, "interval_minutes": 5},
        "quantile_levels": [0.1, 0.5, 0.9],
        "thresholds": {"low_mg_dl": 70, "high_mg_dl": 180},
        "clinic_timezone": "America/Bogota",
        "night_window": {"start": "00:00", "end": "06:00"},
    }
    if mode == "SHORT_TERM":
        body["horizons_minutes"] = [30, 60, 120]
    return body


def falling(n: int = 40, start: float = 260.0, step: float = -6.0) -> list[float]:
    return [round(start + step * i, 1) for i in range(n)]


def risk_body(**features: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "tir_pct": 48.2,
        "tbr_l1_pct": 5.9,
        "tbr_l2_pct": 1.8,
        "tar_l1_pct": 30.1,
        "tar_l2_pct": 14.0,
        "mean_glucose_mg_dl": 176.0,
        "gmi_pct": 7.5,
        "cv_pct": 41.3,
        "data_sufficiency_pct": 92.0,
        "hypo_events_count": 6,
        "severe_hypo_events_count": 0,
        "last_hba1c_pct": 8.1,
        "hba1c_delta_pct": 0.6,
        "tir_delta_vs_prior_pct": -17.0,
        "readings_per_day_avg": 270,
        "days_since_last_reading": 0,
        "open_alerts_count": 3,
        "target_tir_pct": 70,
    }
    base.update(features)
    return {
        "subject_ref": SUBJECT,
        "assessment_type": "DETERIORATION",
        "window_days": 14,
        "diabetes_type": "T1D",
        "features": base,
    }


def summary_body() -> dict[str, Any]:
    """What the backend sends today (wire.summary_body): nulls included."""
    return {
        "subject_ref": SUBJECT,
        "period": {
            "from": "2026-11-01T00:00:00-05:00",
            "to": "2026-11-15T00:00:00-05:00",
        },
        "language": "es",
        "diabetes_type": "T1D",
        "metrics": {
            "tir_pct": 48.2,
            "tbr_l1_pct": 5.9,
            "tbr_l2_pct": 1.8,
            "tar_l1_pct": 30.1,
            "tar_l2_pct": 14.0,
            "mean_glucose_mg_dl": 176.0,
            "gmi_pct": 7.5,
            "cv_pct": 41.3,
            "data_sufficiency_pct": 92.0,
            "agp_percentiles": [],
        },
        "targets": {"tir_target_pct": 70.0},
        "events": [],
        "adherence": {"readings_per_day_avg": 270.0, "dose_logging_pct": None},
        "plan_changes": [],
        "last_hba1c": None,
    }


def assistant_body(message: str, docs: bool = True) -> dict[str, Any]:
    return {
        "subject_ref": SUBJECT,
        "message": message,
        "history": [],
        "context_documents": [
            {
                "doc_id": "doc-hipo-01",
                "title": "Qué es la hipoglucemia",
                "text": "La hipoglucemia es una glucosa baja. Mide tu glucosa y avisa.",
            }
        ]
        if docs
        else [],
        "language": "es",
    }


def llm_json(**value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
