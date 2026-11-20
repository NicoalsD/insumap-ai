"""Request and response models of the v1 contract (Contrato-IA 4, 8 and 9).

Requests reject unknown fields (``extra="forbid"``). Limits that depend on the
settings (points, documents, message length) are checked by the pipelines.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTRACT_VERSION: Final = "1.0.0"
SUBJECT_REF_PATTERN: Final = r"^sub_[0-9a-f]{32}$"
HH_MM: Final = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
EXPLANATION_SUMMARY_MAX: Final = 500
TEXT_MAX: Final = 4000

Percent = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Response(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)


class FallbackReason(StrEnum):
    AI_ERROR = "AI_ERROR"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class Direction(StrEnum):
    INCREASES_RISK = "INCREASES_RISK"
    DECREASES_RISK = "DECREASES_RISK"
    NEUTRAL = "NEUTRAL"


class DataRefKind(StrEnum):
    WINDOW = "WINDOW"
    READINGS = "READINGS"
    DOSES = "DOSES"
    METRIC = "METRIC"


class Factor(Response):
    name: str
    direction: Direction
    value: float | str | None
    unit: str | None
    contribution: float = Field(ge=0, le=1)


class DataRef(Response):
    kind: DataRefKind
    from_: datetime = Field(alias="from")
    to: datetime
    count: int = Field(ge=0)


class Explanation(Response):
    summary: str = Field(max_length=EXPLANATION_SUMMARY_MAX)
    factors: list[Factor] = Field(default_factory=list)
    data_refs: list[DataRef] = Field(default_factory=list)


class Output(Response):
    """Common fields of every ``200`` (Contrato-IA 4.3)."""

    request_id: str
    model_name: str
    model_version: str
    input_hash: str
    generated_at: datetime
    is_fallback: bool
    fallback_reason: FallbackReason | None
    explanation: Explanation


# --- IA1: glucose forecasts (8.1) -------------------------------------------


class ForecastMode(StrEnum):
    SHORT_TERM = "SHORT_TERM"
    OVERNIGHT = "OVERNIGHT"


class Series(Request):
    timestamps: list[str]
    values_mg_dl: list[float]
    interval_minutes: Literal[5, 15]


class Thresholds(Request):
    low_mg_dl: float = Field(gt=0, allow_inf_nan=False)
    high_mg_dl: float = Field(gt=0, allow_inf_nan=False)


class NightWindow(Request):
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def is_hh_mm(cls, value: str) -> str:
        if not HH_MM.match(value):
            raise ValueError("expected HH:MM")
        return value


class GlucoseForecastRequest(Request):
    subject_ref: str = Field(pattern=SUBJECT_REF_PATTERN)
    mode: ForecastMode
    series: Series
    horizons_minutes: list[Annotated[int, Field(ge=5, le=120)]] | None = Field(
        default=None, min_length=1, max_length=6
    )
    quantile_levels: list[float]
    thresholds: Thresholds
    clinic_timezone: str = Field(min_length=1, max_length=64)
    night_window: NightWindow

    @field_validator("quantile_levels")
    @classmethod
    def v1_levels(cls, value: list[float]) -> list[float]:
        if value != [0.1, 0.5, 0.9]:
            raise ValueError("v1 accepts exactly [0.1, 0.5, 0.9]")
        return value


class Trend(StrEnum):
    FALLING = "FALLING"
    STABLE = "STABLE"
    RISING = "RISING"


class ForecastPoint(Response):
    horizon_minutes: int
    timestamp: datetime
    p10: float
    p50: float
    p90: float


class Probability(Response):
    horizon_minutes: int
    probability: float = Field(ge=0, le=1)


class ForecastRisk(Response):
    hypo_probability: list[Probability] = Field(default_factory=list)
    hyper_probability: list[Probability] = Field(default_factory=list)


class GlucoseForecastResponse(Output):
    trend: Trend | None
    points: list[ForecastPoint]
    risk: ForecastRisk


# --- IA2: risk assessments (8.2) --------------------------------------------


class AssessmentType(StrEnum):
    DETERIORATION = "DETERIORATION"
    DROPOUT = "DROPOUT"


class DiabetesType(StrEnum):
    T1D = "T1D"
    T2D = "T2D"
    GESTATIONAL = "GESTATIONAL"


class RiskFeatures(Request):
    tir_pct: Percent | None = None
    tbr_l1_pct: Percent | None = None
    tbr_l2_pct: Percent | None = None
    tar_l1_pct: Percent | None = None
    tar_l2_pct: Percent | None = None
    mean_glucose_mg_dl: NonNegative | None = None
    gmi_pct: Percent | None = None
    cv_pct: NonNegative | None = None
    data_sufficiency_pct: Percent | None = None
    hypo_events_count: int | None = Field(default=None, ge=0)
    severe_hypo_events_count: int | None = Field(default=None, ge=0)
    last_hba1c_pct: Percent | None = None
    hba1c_delta_pct: float | None = Field(default=None, allow_inf_nan=False)
    tir_delta_vs_prior_pct: float | None = Field(default=None, ge=-100, le=100)
    readings_per_day_avg: NonNegative | None = None
    days_since_last_reading: int | None = Field(default=None, ge=0)
    open_alerts_count: int | None = Field(default=None, ge=0)
    target_tir_pct: Percent | None = None
    # DROPOUT (IA7)
    weekly_reading_counts: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, max_length=8
    )
    dose_logging_pct: Percent | None = None
    app_opens_last_14d: int | None = Field(default=None, ge=0)
    unanswered_messages_count: int | None = Field(default=None, ge=0)
    account_age_days: int | None = Field(default=None, ge=0)


class RiskTier(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAssessmentRequest(Request):
    subject_ref: str = Field(pattern=SUBJECT_REF_PATTERN)
    assessment_type: AssessmentType
    window_days: int = Field(ge=7, le=90)
    diabetes_type: DiabetesType
    features: RiskFeatures


class RiskAssessmentResponse(Output):
    assessment_type: AssessmentType
    risk_probability: float | None = Field(ge=0, le=1)
    calibrated: bool
    risk_tier: RiskTier | None


# --- IA5: clinical summaries (8.5) ------------------------------------------


class Period(Request):
    from_: datetime = Field(alias="from")
    to: datetime


class AgpPercentile(Request):
    hour: int = Field(ge=0, le=23)
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class SummaryMetrics(Request):
    tir_pct: Percent | None = None
    tbr_l1_pct: Percent | None = None
    tbr_l2_pct: Percent | None = None
    tar_l1_pct: Percent | None = None
    tar_l2_pct: Percent | None = None
    mean_glucose_mg_dl: NonNegative | None = None
    gmi_pct: Percent | None = None
    cv_pct: NonNegative | None = None
    data_sufficiency_pct: Percent | None = None
    agp_percentiles: list[AgpPercentile] = Field(default_factory=list, max_length=24)


class SummaryTargets(Request):
    tir_target_pct: Percent | None = None
    tbr_l1_target_pct: Percent | None = None
    tbr_l2_target_pct: Percent | None = None
    tar_l1_target_pct: Percent | None = None
    tar_l2_target_pct: Percent | None = None
    cv_target_pct: NonNegative | None = None


class EventKind(StrEnum):
    HYPOGLYCEMIA = "HYPOGLYCEMIA"
    PATTERN = "PATTERN"
    ALERT = "ALERT"
    SYMPTOM = "SYMPTOM"


class SummaryEvent(Request):
    ref_id: str = Field(min_length=1, max_length=64)
    kind: EventKind
    timestamp: datetime
    detail: str = Field(max_length=280)


class Adherence(Request):
    readings_per_day_avg: NonNegative | None = None
    dose_logging_pct: Percent | None = None


class PlanChange(Request):
    ref_id: str = Field(min_length=1, max_length=64)
    date: str = Field(max_length=32)
    description: str = Field(max_length=280)


class Hba1c(Request):
    value_pct: Percent
    date: str = Field(max_length=32)


class ClinicalSummaryRequest(Request):
    subject_ref: str = Field(pattern=SUBJECT_REF_PATTERN)
    period: Period
    language: Literal["es"]
    diabetes_type: DiabetesType
    metrics: SummaryMetrics
    targets: SummaryTargets
    events: list[SummaryEvent] = Field(default_factory=list, max_length=200)
    adherence: Adherence
    plan_changes: list[PlanChange] = Field(default_factory=list, max_length=50)
    last_hba1c: Hba1c | None = None


class SummarySection(Response):
    title: str
    text: str
    refs: list[str] = Field(default_factory=list)


class NumericCheck(Response):
    numbers_in_text: int
    numbers_matched: int
    all_matched: bool


class ClinicalSummaryResponse(Output):
    summary_text: str
    sections: list[SummarySection]
    numeric_check: NumericCheck


# --- IA6: assistant replies (8.6) -------------------------------------------


class Role(StrEnum):
    PATIENT = "PATIENT"
    ASSISTANT = "ASSISTANT"


class Turn(Request):
    role: Role
    text: str = Field(min_length=1, max_length=TEXT_MAX)


class ContextDocument(Request):
    doc_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1)


class AssistantReplyRequest(Request):
    subject_ref: str = Field(pattern=SUBJECT_REF_PATTERN)
    message: str = Field(min_length=1)
    history: list[Turn] = Field(default_factory=list)
    context_documents: list[ContextDocument] = Field(default_factory=list)
    language: Literal["es"]


class EscalationLevel(StrEnum):
    NONE = "NONE"
    CONTACT_DOCTOR = "CONTACT_DOCTOR"
    EMERGENCY = "EMERGENCY"


class Escalation(Response):
    level: EscalationLevel
    message: str | None


class AssistantReplyResponse(Output):
    reply_text: str
    citations: list[str]
    escalation: Escalation
    guardrails_triggered: list[str]


# --- Health and models (9) ---------------------------------------------------


class ModelStatus(StrEnum):
    READY = "READY"
    LOADING = "LOADING"
    FAILED = "FAILED"


class ModelCard(Response):
    name: str
    version: str
    capability: str
    status: ModelStatus
    loaded_at: datetime
    risk_tier_cutoffs: dict[str, float] | None = None


class ModelsResponse(Response):
    contract_version: str
    provider_family: str
    models: list[ModelCard]
