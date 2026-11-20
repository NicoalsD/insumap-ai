"""IA2 and IA7 with the logistic model card."""

from __future__ import annotations

from insumap_ai.contract import RiskAssessmentRequest
from insumap_ai.pipeline import RiskPipeline
from insumap_ai.providers.risk import LogisticRiskModel, load_card
from tests.conftest import risk_body, settings


def run(body: dict) -> object:  # type: ignore[type-arg]
    model = LogisticRiskModel(load_card())
    return RiskPipeline(settings(), model).run(
        RiskAssessmentRequest.model_validate(body), "r"
    )


def test_a_patient_in_range_is_low_risk() -> None:
    result = run(
        risk_body(
            tir_pct=82,
            tbr_l1_pct=1.0,
            tbr_l2_pct=0.2,
            tar_l2_pct=2,
            cv_pct=28,
            tir_delta_vs_prior_pct=3,
            hypo_events_count=0,
            hba1c_delta_pct=-0.2,
            open_alerts_count=0,
        )
    )
    assert result.risk_tier == "LOW"  # type: ignore[attr-defined]
    assert result.is_fallback is False  # type: ignore[attr-defined]


def test_worse_features_raise_the_risk() -> None:
    good = run(risk_body(tir_pct=75)).risk_probability  # type: ignore[attr-defined]
    bad = run(risk_body(tir_pct=30)).risk_probability  # type: ignore[attr-defined]
    assert bad > good


def test_low_data_sufficiency_is_insufficient_data() -> None:
    result = run(risk_body(data_sufficiency_pct=50))
    assert result.is_fallback is True  # type: ignore[attr-defined]
    assert result.fallback_reason == "INSUFFICIENT_DATA"  # type: ignore[attr-defined]
    assert result.risk_probability is None  # type: ignore[attr-defined]


def test_factors_are_ranked_and_normalized() -> None:
    result = run(risk_body())
    factors = result.explanation.factors  # type: ignore[attr-defined]
    assert len(factors) == 4
    contributions = [f.contribution for f in factors]
    assert contributions == sorted(contributions, reverse=True)


def test_dropout() -> None:
    body = {
        "subject_ref": risk_body()["subject_ref"],
        "assessment_type": "DROPOUT",
        "window_days": 56,
        "diabetes_type": "T2D",
        "features": {
            "weekly_reading_counts": [90, 85, 88, 80, 60, 40, 10, 5],
            "days_since_last_reading": 9,
            "dose_logging_pct": 20,
            "app_opens_last_14d": 1,
            "unanswered_messages_count": 4,
        },
    }
    result = run(body)
    assert result.assessment_type == "DROPOUT"  # type: ignore[attr-defined]
    assert result.risk_tier in {"HIGH", "CRITICAL"}  # type: ignore[attr-defined]
    assert "abandono" in result.explanation.summary  # type: ignore[attr-defined]
    empty = {**body, "features": {}}
    assert run(empty).fallback_reason == "INSUFFICIENT_DATA"  # type: ignore[attr-defined]


def test_model_failure_falls_back() -> None:
    class Broken(LogisticRiskModel):
        def deterioration(self, features):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    result = RiskPipeline(settings(), Broken(load_card())).run(
        RiskAssessmentRequest.model_validate(risk_body()), "r"
    )
    assert result.fallback_reason == "AI_ERROR"
    assert result.model_version == "fallback-rules@1.0.0"
