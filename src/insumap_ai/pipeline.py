"""The inference pipelines of IA1, IA2/IA7, IA5 and IA6.

Every pipeline follows the same steps (``InferencePipeline.run``): check the
technical limits, infer with the configured family, and on any model failure
answer with the deterministic fallback and ``is_fallback = true``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any, Final, override
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ValidationError

from insumap_ai.config import Settings
from insumap_ai.contract import (
    AssessmentType,
    AssistantReplyRequest,
    AssistantReplyResponse,
    ClinicalSummaryRequest,
    ClinicalSummaryResponse,
    DataRef,
    DataRefKind,
    Direction,
    Escalation,
    EscalationLevel,
    Explanation,
    Factor,
    FallbackReason,
    ForecastMode,
    ForecastPoint,
    ForecastRisk,
    GlucoseForecastRequest,
    GlucoseForecastResponse,
    Probability,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    RiskTier,
    Role,
    SummarySection,
    Trend,
)
from insumap_ai.errors import InvalidSeriesError, LimitExceededError
from insumap_ai.guardrails import (
    CONTACT_DOCTOR_MESSAGE,
    EMERGENCY_MESSAGE,
    FALLBACK_REPLY,
    OFF_TOPIC_REPLY,
    GuardContext,
    Guardrail,
    input_chain,
    output_chain,
)
from insumap_ai.prompts import ASSISTANT_SYSTEM, SUMMARY_SYSTEM, PromptBuilder
from insumap_ai.providers.base import (
    ChatRole,
    Forecaster,
    ForecastResult,
    GlucoseSeries,
    RiskModel,
    RiskScore,
    TextGenerationError,
    TextGenerator,
)
from insumap_ai.redaction import PhiRedactor
from insumap_ai.summary import allowed_figures, numeric_check, template_sections

logger = logging.getLogger("insumap_ai.pipeline")

SUMMARY_MAX: Final = 500
TEXT_MAX: Final = 4000
OVERNIGHT_STEP_MINUTES: Final = 60
OVERNIGHT_MAX_MINUTES: Final = 720
FACTORS_SHOWN: Final = 4
SENTENCE_END: Final = re.compile(r"[.!?](\s|$)")
MARKDOWN: Final = re.compile(r"(\*\*|__|`|^#+\s*|^\s*[-*]\s+)", re.MULTILINE)


class InsufficientDataError(Exception):
    """Not enough data for the model: a 200 with ``INSUFFICIENT_DATA``."""


class ModelOutputRejectedError(Exception):
    """The model answered, but the answer failed a check of the service."""


@dataclass(frozen=True, slots=True)
class Meta:
    request_id: str
    input_hash: str
    generated_at: datetime


def input_hash(request: BaseModel) -> str:
    """SHA-256 of the canonical JSON of the request (Contrato-IA 4.3)."""
    body = request.model_dump(mode="json", by_alias=True)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def plain(text: str) -> str:
    """The LLM output as plain text: no Markdown marks, collapsed spaces."""
    return re.sub(r"[ \t]+", " ", MARKDOWN.sub("", text)).strip()


def clip_sentences(text: str, limit: int) -> str:
    """Cut at the last full sentence that fits in ``limit`` characters."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in SENTENCE_END.finditer(head)]
    return head[: ends[-1]].strip() if ends else head.rstrip()


def short(text: str) -> str:
    return clip_sentences(text, SUMMARY_MAX)


def parse_json_object(raw: str) -> dict[str, Any]:
    """The JSON object of an LLM reply, tolerating a Markdown code fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ModelOutputRejectedError("NOT_JSON") from error
    if not isinstance(value, dict):
        raise ModelOutputRejectedError("NOT_OBJECT")
    return value


class InferencePipeline[Req: BaseModel, Resp: BaseModel](ABC):
    """Validate, infer, fall back: the fixed skeleton of every use case.

    Pattern: P20 Template Method
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, request: Req, request_id: str) -> Resp:
        self.validate(request)
        meta = Meta(
            request_id=request_id,
            input_hash=input_hash(request),
            generated_at=datetime.now(UTC),
        )
        try:
            return self.infer(request, meta)
        except InsufficientDataError:
            return self.fallback(request, meta, FallbackReason.INSUFFICIENT_DATA)
        except Exception as error:  # any model failure falls back
            logger.warning(
                "model failure, answering with the fallback",
                extra={
                    "request_id": request_id,
                    "pipeline": type(self).__name__,
                    "error": type(error).__name__,
                    "reason": str(error)[:80]
                    if isinstance(error, TextGenerationError | ModelOutputRejectedError)
                    else "",
                },
            )
            return self.fallback(request, meta, FallbackReason.AI_ERROR)

    def validate(self, request: Req) -> None:  # noqa: B027 - optional hook
        """Technical limits of the settings; raises a ``ServiceError``."""

    @abstractmethod
    def infer(self, request: Req, meta: Meta) -> Resp: ...

    @abstractmethod
    def fallback(self, request: Req, meta: Meta, reason: FallbackReason) -> Resp: ...


# --- IA1 ----------------------------------------------------------------------


class ForecastPipeline(
    InferencePipeline[GlucoseForecastRequest, GlucoseForecastResponse]
):
    def __init__(
        self, settings: Settings, forecaster: Forecaster, fallback: Forecaster
    ) -> None:
        super().__init__(settings)
        self._forecaster = forecaster
        self._fallback = fallback

    @override
    def validate(self, request: GlucoseForecastRequest) -> None:
        series = request.series
        s = self._settings
        if len(series.timestamps) != len(series.values_mg_dl):
            raise InvalidSeriesError("LENGTH_MISMATCH")
        if len(series.timestamps) > s.max_forecast_points:
            raise InvalidSeriesError("TOO_MANY_POINTS")
        moments = []
        for raw in series.timestamps:
            try:
                moment = datetime.fromisoformat(raw)
            except ValueError as error:
                raise InvalidSeriesError("TIMESTAMP_PARSE_ERROR") from error
            if moment.tzinfo is None:
                raise InvalidSeriesError("TIMESTAMP_PARSE_ERROR")
            moments.append(moment)
        if any(b <= a for a, b in itertools.pairwise(moments)):
            raise InvalidSeriesError("TIMESTAMPS_NOT_INCREASING")
        for value in series.values_mg_dl:
            if not math.isfinite(value):
                raise InvalidSeriesError("NOT_FINITE_VALUE")
            if not s.glucose_plausible_min <= value <= s.glucose_plausible_max:
                raise InvalidSeriesError("VALUE_OUT_OF_RANGE")
        if request.mode is ForecastMode.SHORT_TERM and not request.horizons_minutes:
            raise LimitExceededError("horizons_minutes", "REQUIRED_FOR_SHORT_TERM")
        try:
            ZoneInfo(request.clinic_timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise LimitExceededError("clinic_timezone", "UNKNOWN_TIMEZONE") from error

    @override
    def infer(
        self, request: GlucoseForecastRequest, meta: Meta
    ) -> GlucoseForecastResponse:
        series = self._series(request)
        if len(series.values) < self._settings.forecast_min_points:
            raise InsufficientDataError
        horizons = self._horizons(request, series)
        result = self._forecaster.forecast(series, horizons)
        return self._respond(request, meta, series, result, self._forecaster, None)

    @override
    def fallback(
        self,
        request: GlucoseForecastRequest,
        meta: Meta,
        reason: FallbackReason,
    ) -> GlucoseForecastResponse:
        series = self._series(request)
        if len(series.values) < 2:  # noqa: PLR2004 - a slope needs two points
            return GlucoseForecastResponse(
                request_id=meta.request_id,
                model_name=self._fallback.info.name,
                model_version=self._fallback.info.version,
                input_hash=meta.input_hash,
                generated_at=meta.generated_at,
                is_fallback=True,
                fallback_reason=reason,
                trend=None,
                points=[],
                risk=ForecastRisk(),
                explanation=Explanation(
                    summary="No hay lecturas suficientes para pronosticar."
                ),
            )
        horizons = self._horizons(request, series)
        result = self._fallback.forecast(series, horizons)
        return self._respond(request, meta, series, result, self._fallback, reason)

    def _series(self, request: GlucoseForecastRequest) -> GlucoseSeries:
        return GlucoseSeries(
            timestamps=tuple(
                datetime.fromisoformat(raw) for raw in request.series.timestamps
            ),
            values=tuple(request.series.values_mg_dl),
            interval_minutes=request.series.interval_minutes,
        )

    def _horizons(
        self, request: GlucoseForecastRequest, series: GlucoseSeries
    ) -> tuple[int, ...]:
        if request.mode is ForecastMode.SHORT_TERM:
            return tuple(sorted(set(request.horizons_minutes or ())))
        zone = ZoneInfo(request.clinic_timezone)
        last = series.timestamps[-1].astimezone(zone)
        hour, minute = (int(part) for part in request.night_window.end.split(":"))
        end = datetime.combine(last.date(), time(hour, minute), tzinfo=zone)
        if end <= last:
            end += timedelta(days=1)
        total = min(OVERNIGHT_MAX_MINUTES, int((end - last).total_seconds() // 60))
        steps = list(range(OVERNIGHT_STEP_MINUTES, total, OVERNIGHT_STEP_MINUTES))
        return tuple([*steps, total]) if total > 0 else (OVERNIGHT_STEP_MINUTES,)

    def _respond(
        self,
        request: GlucoseForecastRequest,
        meta: Meta,
        series: GlucoseSeries,
        result: ForecastResult,
        model: Forecaster,
        reason: FallbackReason | None,
    ) -> GlucoseForecastResponse:
        last = series.timestamps[-1]
        low, high = request.thresholds.low_mg_dl, request.thresholds.high_mg_dl
        points, hypo, hyper = [], [], []
        for point in result.points:
            points.append(
                ForecastPoint(
                    horizon_minutes=point.horizon_minutes,
                    timestamp=last + timedelta(minutes=point.horizon_minutes),
                    p10=point.p10,
                    p50=point.p50,
                    p90=point.p90,
                )
            )
            hypo.append(
                Probability(
                    horizon_minutes=point.horizon_minutes,
                    probability=_below(low, point.p50, point.sigma),
                )
            )
            hyper.append(
                Probability(
                    horizon_minutes=point.horizon_minutes,
                    probability=round(1 - _below(high, point.p50, point.sigma), 3),
                )
            )
        threshold = self._settings.forecast_trend_threshold_mg_dl_min
        slope = result.slope_mg_dl_min
        trend = (
            Trend.FALLING
            if slope <= -threshold
            else Trend.RISING
            if slope >= threshold
            else Trend.STABLE
        )
        window = timedelta(minutes=self._settings.forecast_slope_window_minutes)
        recent = [t for t in series.timestamps if t >= last - window]
        head = points[0]
        summary = (
            f"Glucosa {_trend_es(trend)} a {_n(slope)} mg/dL por minuto; última "
            f"lectura {_n(series.values[-1])} mg/dL. En {head.horizon_minutes} minutos "
            f"el rango probable es {_n(head.p10)} a {_n(head.p90)} mg/dL."
        )
        return GlucoseForecastResponse(
            request_id=meta.request_id,
            model_name=model.info.name,
            model_version=model.info.version,
            input_hash=meta.input_hash,
            generated_at=meta.generated_at,
            is_fallback=reason is not None,
            fallback_reason=reason,
            trend=trend,
            points=points,
            risk=ForecastRisk(hypo_probability=hypo, hyper_probability=hyper),
            explanation=Explanation(
                summary=short(summary),
                factors=[
                    Factor(
                        name="recent_slope",
                        direction=Direction.INCREASES_RISK
                        if trend is not Trend.STABLE
                        else Direction.NEUTRAL,
                        value=round(slope, 2),
                        unit="mg/dL/min",
                        contribution=0.7,
                    ),
                    Factor(
                        name="latest_value",
                        direction=Direction.NEUTRAL,
                        value=series.values[-1],
                        unit="mg/dL",
                        contribution=0.3,
                    ),
                ],
                data_refs=[
                    DataRef(
                        kind=DataRefKind.READINGS,
                        from_=recent[0],
                        to=last,
                        count=len(recent),
                    )
                ],
            ),
        )


def _below(threshold: float, mean: float, sigma: float) -> float:
    """P(X < threshold) for a normal spread; a step when there is no spread."""
    if sigma <= 0:
        return 1.0 if mean < threshold else 0.0
    z = (threshold - mean) / (sigma * math.sqrt(2))
    return round(0.5 * (1 + math.erf(z)), 3)


def _trend_es(trend: Trend) -> str:
    return {
        Trend.FALLING: "en descenso",
        Trend.RISING: "en ascenso",
        Trend.STABLE: "estable",
    }[trend]


def _n(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".").replace(".", ",")


# --- IA2 and IA7 ----------------------------------------------------------------

FEATURE_LABELS: Final = {
    "tir_gap_pct": "TIR por debajo de la meta",
    "tbr_l1_pct": "tiempo bajo rango nivel 1",
    "tbr_l2_pct": "tiempo bajo rango nivel 2",
    "tar_l2_pct": "tiempo sobre rango nivel 2",
    "cv_pct": "variabilidad (CV)",
    "tir_drop_pct": "caída del TIR frente al periodo anterior",
    "hypo_events_count": "episodios de hipoglucemia",
    "severe_hypo_events_count": "hipoglucemias graves",
    "hba1c_delta_pct": "cambio de HbA1c",
    "open_alerts_count": "alertas abiertas",
    "days_since_last_reading": "días sin lecturas",
    "weekly_reading_decline_pct": "caída de lecturas semanales",
    "dose_logging_gap_pct": "dosis sin registrar",
    "app_opens_last_14d": "uso de la app",
    "unanswered_messages_count": "mensajes sin responder",
}
TIER_ES: Final = {
    RiskTier.LOW: "bajo",
    RiskTier.MEDIUM: "medio",
    RiskTier.HIGH: "alto",
    RiskTier.CRITICAL: "crítico",
}
RECENT_WEEKS: Final = 2


class RiskPipeline(InferencePipeline[RiskAssessmentRequest, RiskAssessmentResponse]):
    def __init__(self, settings: Settings, model: RiskModel) -> None:
        super().__init__(settings)
        self._model = model

    @override
    def infer(
        self, request: RiskAssessmentRequest, meta: Meta
    ) -> RiskAssessmentResponse:
        f = request.features
        if request.assessment_type is AssessmentType.DETERIORATION:
            sufficiency = f.data_sufficiency_pct
            if (
                sufficiency is None
                or sufficiency < self._settings.risk_min_data_sufficiency_pct
            ):
                raise InsufficientDataError
            score = self._model.deterioration(_deterioration_features(request))
        else:
            features = _dropout_features(request)
            if all(value is None for value in features.values()):
                raise InsufficientDataError
            score = self._model.dropout(features)
        tier = self._tier(score.probability)
        return RiskAssessmentResponse(
            request_id=meta.request_id,
            model_name=self._model.info.name,
            model_version=self._model.info.version,
            input_hash=meta.input_hash,
            generated_at=meta.generated_at,
            is_fallback=False,
            fallback_reason=None,
            assessment_type=request.assessment_type,
            risk_probability=round(score.probability, 3),
            calibrated=self._model.calibrated,
            risk_tier=tier,
            explanation=_risk_explanation(request, score, tier),
        )

    @override
    def fallback(
        self, request: RiskAssessmentRequest, meta: Meta, reason: FallbackReason
    ) -> RiskAssessmentResponse:
        summary = (
            "Datos insuficientes para estimar el riesgo; el backend aplica sus reglas."
            if reason is FallbackReason.INSUFFICIENT_DATA
            else "El modelo de riesgo no respondió; el backend aplica sus reglas."
        )
        return RiskAssessmentResponse(
            request_id=meta.request_id,
            model_name="risk-rules-fallback",
            model_version="fallback-rules@1.0.0",
            input_hash=meta.input_hash,
            generated_at=meta.generated_at,
            is_fallback=True,
            fallback_reason=reason,
            assessment_type=request.assessment_type,
            risk_probability=None,
            calibrated=False,
            risk_tier=None,
            explanation=Explanation(summary=summary),
        )

    def _tier(self, probability: float) -> RiskTier:
        s = self._settings
        if probability >= s.risk_tier_critical:
            return RiskTier.CRITICAL
        if probability >= s.risk_tier_high:
            return RiskTier.HIGH
        if probability >= s.risk_tier_medium:
            return RiskTier.MEDIUM
        return RiskTier.LOW


def _deterioration_features(request: RiskAssessmentRequest) -> dict[str, float | None]:
    f = request.features
    gap = (
        None
        if f.tir_pct is None or f.target_tir_pct is None
        else f.target_tir_pct - f.tir_pct
    )
    drop = None if f.tir_delta_vs_prior_pct is None else -f.tir_delta_vs_prior_pct
    return {
        "tir_gap_pct": gap,
        "tbr_l1_pct": f.tbr_l1_pct,
        "tbr_l2_pct": f.tbr_l2_pct,
        "tar_l2_pct": f.tar_l2_pct,
        "cv_pct": f.cv_pct,
        "tir_drop_pct": drop,
        "hypo_events_count": _float(f.hypo_events_count),
        "severe_hypo_events_count": _float(f.severe_hypo_events_count),
        "hba1c_delta_pct": f.hba1c_delta_pct,
        "open_alerts_count": _float(f.open_alerts_count),
    }


def _dropout_features(request: RiskAssessmentRequest) -> dict[str, float | None]:
    f = request.features
    decline = None
    counts = f.weekly_reading_counts
    if counts and len(counts) > RECENT_WEEKS:
        before = sum(counts[:-RECENT_WEEKS]) / len(counts[:-RECENT_WEEKS])
        recent = sum(counts[-RECENT_WEEKS:]) / RECENT_WEEKS
        if before > 0:
            decline = (before - recent) / before * 100
    return {
        "days_since_last_reading": _float(f.days_since_last_reading),
        "weekly_reading_decline_pct": decline,
        "dose_logging_gap_pct": None
        if f.dose_logging_pct is None
        else 100 - f.dose_logging_pct,
        "app_opens_last_14d": _float(f.app_opens_last_14d),
        "unanswered_messages_count": _float(f.unanswered_messages_count),
    }


def _float(value: int | None) -> float | None:
    return None if value is None else float(value)


def _risk_explanation(
    request: RiskAssessmentRequest, score: RiskScore, tier: RiskTier
) -> Explanation:
    total = sum(abs(c.weight) for c in score.contributions) or 1.0
    ranked = sorted(score.contributions, key=lambda c: abs(c.weight), reverse=True)
    factors = [
        Factor(
            name=c.name,
            direction=Direction.INCREASES_RISK
            if c.weight > 0
            else Direction.DECREASES_RISK
            if c.weight < 0
            else Direction.NEUTRAL,
            value=None if c.value is None else round(float(c.value), 2),
            unit=c.unit,
            contribution=round(abs(c.weight) / total, 3),
        )
        for c in ranked[:FACTORS_SHOWN]
    ]
    drivers = [FEATURE_LABELS.get(c.name, c.name) for c in ranked if c.weight > 0][:2]
    kind = (
        "de deterioro"
        if request.assessment_type is AssessmentType.DETERIORATION
        else "de abandono"
    )
    because = f": pesan {' y '.join(drivers)}" if drivers else ""
    return Explanation(
        summary=short(f"Riesgo {kind} {TIER_ES[tier]}{because}. Puntaje sin calibrar."),
        factors=factors,
    )


# --- IA5 ------------------------------------------------------------------------


class SummaryPipeline(
    InferencePipeline[ClinicalSummaryRequest, ClinicalSummaryResponse]
):
    def __init__(self, settings: Settings, text: TextGenerator) -> None:
        super().__init__(settings)
        self._text = text

    @override
    def infer(
        self, request: ClinicalSummaryRequest, meta: Meta
    ) -> ClinicalSummaryResponse:
        payload = request.model_dump(
            mode="json", by_alias=True, exclude={"subject_ref"}
        )
        prompt = (
            PromptBuilder()
            .system(SUMMARY_SYSTEM)
            .turn(ChatRole.USER, json.dumps(payload, ensure_ascii=False))
            .build()
        )
        reply = parse_json_object(self._text.generate(prompt, json_output=True))
        figures = allowed_figures(request)
        sections = _sections(reply, _known_refs(request, figures))
        text = clip_sentences("\n\n".join(s.text for s in sections), TEXT_MAX)
        check = numeric_check(text, figures.values())
        if not check.all_matched:
            raise ModelOutputRejectedError("NUMERIC_MISMATCH")
        return ClinicalSummaryResponse(
            request_id=meta.request_id,
            model_name=self._text.info.name,
            model_version=self._text.info.version,
            input_hash=meta.input_hash,
            generated_at=meta.generated_at,
            is_fallback=False,
            fallback_reason=None,
            summary_text=text,
            sections=sections,
            numeric_check=check,
            explanation=Explanation(
                summary=(
                    "Resumen generado solo con las cifras recibidas; todas las cifras "
                    "coinciden con la entrada."
                ),
                data_refs=[_window(request)],
            ),
        )

    @override
    def fallback(
        self, request: ClinicalSummaryRequest, meta: Meta, reason: FallbackReason
    ) -> ClinicalSummaryResponse:
        sections = template_sections(request)
        text = "\n\n".join(s.text for s in sections)
        return ClinicalSummaryResponse(
            request_id=meta.request_id,
            model_name="clinical-summary-template",
            model_version="template@1.0.0",
            input_hash=meta.input_hash,
            generated_at=meta.generated_at,
            is_fallback=True,
            fallback_reason=reason,
            summary_text=text,
            sections=sections,
            numeric_check=numeric_check(text, allowed_figures(request).values()),
            explanation=Explanation(
                summary="Resumen con plantilla fija a partir de las cifras recibidas.",
                data_refs=[_window(request)],
            ),
        )


def _known_refs(request: ClinicalSummaryRequest, figures: dict[str, float]) -> set[str]:
    refs = set(figures)
    refs.update(e.ref_id for e in request.events)
    refs.update(p.ref_id for p in request.plan_changes)
    return refs


def _sections(reply: dict[str, Any], known: set[str]) -> list[SummarySection]:
    raw = reply.get("sections")
    if not isinstance(raw, list) or not raw:
        raise ModelOutputRejectedError("NO_SECTIONS")
    sections = []
    for item in raw:
        if not isinstance(item, dict):
            raise ModelOutputRejectedError("BAD_SECTION")
        title = plain(str(item.get("title", "")))[:200]
        text = clip_sentences(plain(str(item.get("text", ""))), TEXT_MAX)
        refs = item.get("refs", [])
        if not title or not text:
            raise ModelOutputRejectedError("EMPTY_SECTION")
        clean_refs = [r for r in refs if isinstance(r, str) and r in known]
        sections.append(SummarySection(title=title, text=text, refs=clean_refs))
    return sections


def _window(request: ClinicalSummaryRequest) -> DataRef:
    return DataRef(
        kind=DataRefKind.WINDOW,
        from_=request.period.from_,
        to=request.period.to,
        count=len(request.events),
    )


# --- IA6 ------------------------------------------------------------------------


class _Reply(BaseModel):
    reply: str
    citations: list[str] = []
    on_topic: bool = True


class AssistantPipeline(
    InferencePipeline[AssistantReplyRequest, AssistantReplyResponse]
):
    def __init__(self, settings: Settings, text: TextGenerator) -> None:
        super().__init__(settings)
        self._text = text
        self._redactor = PhiRedactor()

    @override
    def validate(self, request: AssistantReplyRequest) -> None:
        s = self._settings
        if len(request.message) > s.max_message_chars:
            raise LimitExceededError("message", "TOO_LONG")
        if len(request.history) > s.max_history_turns:
            raise LimitExceededError("history", "TOO_MANY_TURNS")
        if len(request.context_documents) > s.max_context_docs:
            raise LimitExceededError("context_documents", "TOO_MANY_DOCUMENTS")
        for index, doc in enumerate(request.context_documents):
            if len(doc.text) > s.max_context_doc_chars:
                raise LimitExceededError(f"context_documents.{index}.text", "TOO_LONG")

    @override
    def infer(
        self, request: AssistantReplyRequest, meta: Meta
    ) -> AssistantReplyResponse:
        guard = input_chain(self._redactor).handle(GuardContext(request.message))
        if guard.blocked_reply is not None:
            return self._blocked(request, meta, guard, guard.blocked_reply)
        try:
            reply = self._ask(request, guard.text)
        except (
            TextGenerationError,
            ModelOutputRejectedError,
            ValidationError,
        ) as error:
            logger.warning(
                "assistant model failure, answering with the fallback",
                extra={"request_id": meta.request_id, "error": type(error).__name__},
            )
            return self._blocked(request, meta, guard, FALLBACK_REPLY)
        if not reply.on_topic:
            guard.trigger(Guardrail.OFF_TOPIC)
            return self._blocked(request, meta, guard, OFF_TOPIC_REPLY)
        checked = output_chain(self._redactor).handle(GuardContext(plain(reply.reply)))
        for name in checked.triggered:
            guard.trigger(name)
        if checked.blocked_reply is not None:
            return self._blocked(request, meta, guard, checked.blocked_reply)
        known = {doc.doc_id: doc.title for doc in request.context_documents}
        citations = list(dict.fromkeys(c for c in reply.citations if c in known))
        text = clip_sentences(checked.text, self._settings.max_message_chars)
        if not text:
            raise ModelOutputRejectedError("EMPTY_REPLY")
        used = ", ".join(f"'{known[c]}'" for c in citations)
        summary = (
            f"Respuesta basada en el material aprobado {used}."
            if citations
            else "Respuesta educativa general: ningún material aprobado cubría la "
            "pregunta."
        )
        return AssistantReplyResponse(
            request_id=meta.request_id,
            model_name=self._text.info.name,
            model_version=self._text.info.version,
            input_hash=meta.input_hash,
            generated_at=meta.generated_at,
            is_fallback=False,
            fallback_reason=None,
            reply_text=text,
            citations=citations,
            escalation=_escalation(guard.escalation),
            guardrails_triggered=[g.value for g in guard.triggered],
            explanation=Explanation(summary=short(summary)),
        )

    @override
    def fallback(
        self, request: AssistantReplyRequest, meta: Meta, reason: FallbackReason
    ) -> AssistantReplyResponse:
        guard = input_chain(self._redactor).handle(GuardContext(request.message))
        return self._blocked(
            request, meta, guard, guard.blocked_reply or FALLBACK_REPLY
        )

    def _ask(self, request: AssistantReplyRequest, message: str) -> _Reply:
        builder = PromptBuilder().system(ASSISTANT_SYSTEM)
        for doc in request.context_documents:
            builder.document(doc.doc_id, doc.title, doc.text)
        for turn in request.history:
            role = ChatRole.USER if turn.role is Role.PATIENT else ChatRole.ASSISTANT
            builder.turn(role, turn.text)
        builder.turn(ChatRole.USER, message)
        raw = parse_json_object(self._text.generate(builder.build(), json_output=True))
        return _Reply.model_validate(raw)

    def _blocked(
        self,
        request: AssistantReplyRequest,
        meta: Meta,
        guard: GuardContext,
        reply: str,
    ) -> AssistantReplyResponse:
        del request
        return AssistantReplyResponse(
            request_id=meta.request_id,
            model_name="assistant-guardrails",
            model_version="guardrails@1.0.0",
            input_hash=meta.input_hash,
            generated_at=meta.generated_at,
            is_fallback=True,
            fallback_reason=FallbackReason.AI_ERROR,
            reply_text=reply,
            citations=[],
            escalation=_escalation(guard.escalation),
            guardrails_triggered=[g.value for g in guard.triggered],
            explanation=Explanation(
                summary="Respuesta predefinida: actuó un guardrail o el modelo no "
                "respondió."
            ),
        )


def _escalation(level: EscalationLevel) -> Escalation:
    message = {
        EscalationLevel.NONE: None,
        EscalationLevel.CONTACT_DOCTOR: CONTACT_DOCTOR_MESSAGE,
        EscalationLevel.EMERGENCY: EMERGENCY_MESSAGE,
    }[level]
    return Escalation(level=level, message=message)
