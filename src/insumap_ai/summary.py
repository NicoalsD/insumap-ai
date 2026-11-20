"""IA5 helpers: the figures a summary may quote, the numeric fidelity check and
the deterministic template."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Final

from insumap_ai.contract import (
    ClinicalSummaryRequest,
    EventKind,
    NumericCheck,
    SummarySection,
)

NUMBER: Final = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")
LEVEL_LABELS: Final = (1.0, 2.0)
""""nivel 1" and "nivel 2" name the TBR and TAR levels; they are not figures."""
TOLERANCE: Final = 1e-9
ROUNDINGS: Final = (0, 1, 2)
SECONDS_PER_DAY: Final = 86400


def allowed_figures(request: ClinicalSummaryRequest) -> dict[str, float]:
    """Every figure of the input, keyed by the ref the summary may cite."""
    figures: dict[str, float] = {}
    for prefix, model in (
        ("metric", request.metrics),
        ("target", request.targets),
        ("adherence", request.adherence),
    ):
        for name, value in model.model_dump().items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                figures[f"{prefix}:{name}"] = float(value)
    for row in request.metrics.agp_percentiles:
        for name in ("p5", "p25", "p50", "p75", "p95"):
            figures[f"metric:agp_{row.hour}_{name}"] = float(getattr(row, name))
    if request.last_hba1c is not None:
        figures["metric:last_hba1c"] = request.last_hba1c.value_pct
    seconds = (request.period.to - request.period.from_).total_seconds()
    days = seconds / SECONDS_PER_DAY
    figures["period:days"] = float(round(days))
    figures["count:events"] = float(len(request.events))
    figures["count:plan_changes"] = float(len(request.plan_changes))
    counts = Counter(event.kind for event in request.events)
    for kind in EventKind:
        figures[f"count:{kind.value.lower()}"] = float(counts.get(kind, 0))
    return figures


def numeric_check(text: str, figures: Iterable[float]) -> NumericCheck:
    """Every number in ``text`` must be one of ``figures`` (decimal comma allowed,
    rounding to 0 to 2 decimals tolerated)."""
    accepted = {round(value, digits) for value in figures for digits in ROUNDINGS}
    accepted.update(LEVEL_LABELS)
    found = [float(raw.replace(",", ".")) for raw in NUMBER.findall(text)]
    matched = sum(
        1 for number in found if any(abs(number - a) < TOLERANCE for a in accepted)
    )
    return NumericCheck(
        numbers_in_text=len(found),
        numbers_matched=matched,
        all_matched=matched == len(found),
    )


def fmt(value: float) -> str:
    """``48.2`` as ``48,2`` and ``70.0`` as ``70``."""
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _join(parts: list[str], empty: str) -> str:
    return " ".join(parts) if parts else empty


def template_sections(request: ClinicalSummaryRequest) -> list[SummarySection]:
    """The deterministic summary: same figures, fixed wording."""
    m, t, a = request.metrics, request.targets, request.adherence
    period: list[str] = []
    refs: list[str] = []
    if m.tir_pct is not None:
        goal = "" if t.tir_target_pct is None else f" (meta {fmt(t.tir_target_pct)} %)"
        period.append(f"TIR {fmt(m.tir_pct)} %{goal}.")
        refs.append("metric:tir_pct")
    for label, value, unit, ref in (
        ("Glucosa media", m.mean_glucose_mg_dl, " mg/dL", "metric:mean_glucose_mg_dl"),
        ("GMI", m.gmi_pct, " %", "metric:gmi_pct"),
        ("CV", m.cv_pct, " %", "metric:cv_pct"),
        (
            "Datos del sensor",
            m.data_sufficiency_pct,
            " %",
            "metric:data_sufficiency_pct",
        ),
    ):
        if value is not None:
            period.append(f"{label} {fmt(value)}{unit}.")
            refs.append(ref)
    if request.last_hba1c is not None:
        period.append(f"Última HbA1c {fmt(request.last_hba1c.value_pct)} %.")
        refs.append("metric:last_hba1c")

    ranges: list[str] = []
    range_refs: list[str] = []
    for label, value, ref in (
        ("Tiempo bajo rango nivel 1", m.tbr_l1_pct, "metric:tbr_l1_pct"),
        ("Tiempo bajo rango nivel 2", m.tbr_l2_pct, "metric:tbr_l2_pct"),
        ("Tiempo sobre rango nivel 1", m.tar_l1_pct, "metric:tar_l1_pct"),
        ("Tiempo sobre rango nivel 2", m.tar_l2_pct, "metric:tar_l2_pct"),
    ):
        if value is not None:
            ranges.append(f"{label}: {fmt(value)} %.")
            range_refs.append(ref)
    hypos = [e for e in request.events if e.kind is EventKind.HYPOGLYCEMIA]
    if hypos:
        ranges.append(f"Episodios de hipoglucemia registrados: {len(hypos)}.")
        range_refs.extend(e.ref_id for e in hypos)

    adherence: list[str] = []
    adherence_refs: list[str] = []
    if a.readings_per_day_avg is not None:
        adherence.append(
            f"Lecturas por día en promedio: {fmt(a.readings_per_day_avg)}."
        )
        adherence_refs.append("adherence:readings_per_day_avg")
    if a.dose_logging_pct is not None:
        adherence.append(f"Dosis registradas: {fmt(a.dose_logging_pct)} %.")
        adherence_refs.append("adherence:dose_logging_pct")

    unavailable = "Sin datos disponibles en el periodo."
    return [
        SummarySection(
            title="Resumen del periodo", text=_join(period, unavailable), refs=refs
        ),
        SummarySection(
            title="Hipoglucemia e hiperglucemia",
            text=_join(ranges, unavailable),
            refs=range_refs,
        ),
        SummarySection(
            title="Adherencia", text=_join(adherence, unavailable), refs=adherence_refs
        ),
    ]
