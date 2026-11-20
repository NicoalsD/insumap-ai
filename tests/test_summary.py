"""IA5: numeric fidelity, refs, the template and the LLM path."""

from __future__ import annotations

from insumap_ai.contract import ClinicalSummaryRequest
from insumap_ai.pipeline import SummaryPipeline
from insumap_ai.providers.base import TextGenerationError
from insumap_ai.summary import allowed_figures, fmt, numeric_check, template_sections
from tests.conftest import ScriptedTextGenerator, llm_json, settings, summary_body


def request() -> ClinicalSummaryRequest:
    return ClinicalSummaryRequest.model_validate(summary_body())


def test_numeric_check_accepts_decimal_comma_and_rounding() -> None:
    figures = [48.2, 70.0, 176.0]
    check = numeric_check("TIR 48,2 % (meta 70 %), media 176 mg/dL, nivel 2", figures)
    assert check.all_matched and check.numbers_in_text == 4


def test_numeric_check_catches_an_invented_figure() -> None:
    check = numeric_check("TIR 52,0 % en 14 días", [48.2])
    assert not check.all_matched
    assert check.numbers_matched == 0


def test_allowed_figures_include_period_and_counts() -> None:
    figures = allowed_figures(request())
    assert figures["period:days"] == 14
    assert figures["metric:tir_pct"] == 48.2
    assert "adherence:dose_logging_pct" not in figures


def test_fmt() -> None:
    assert fmt(48.2) == "48,2" and fmt(70.0) == "70" and fmt(7.55) == "7,5"


def test_template_matches_every_figure() -> None:
    sections = template_sections(request())
    text = " ".join(s.text for s in sections)
    assert numeric_check(text, allowed_figures(request()).values()).all_matched
    assert sections[0].refs[0] == "metric:tir_pct"


def test_llm_summary_is_kept_when_figures_match() -> None:
    reply = llm_json(
        sections=[
            {
                "title": "Resumen del periodo",
                "text": "**TIR 48,2 %**, por debajo de la meta de 70 % en 14 días.",
                "refs": ["metric:tir_pct", "target:tir_target_pct", "invented:ref"],
            },
            {"title": "Variabilidad", "text": "CV 41,3 %.", "refs": ["metric:cv_pct"]},
        ]
    )
    llm = ScriptedTextGenerator(reply)
    result = SummaryPipeline(settings(), llm).run(request(), "rid")
    assert result.is_fallback is False
    assert result.model_name == "scripted-llm"
    assert result.sections[0].text.startswith("TIR 48,2 %")
    assert result.sections[0].refs == ["metric:tir_pct", "target:tir_target_pct"]
    assert result.numeric_check.all_matched
    assert "sub_" not in llm.prompts[0].messages[0].content


def test_llm_summary_with_an_invented_figure_falls_back() -> None:
    reply = llm_json(sections=[{"title": "Resumen", "text": "TIR 60 %.", "refs": []}])
    result = SummaryPipeline(settings(), ScriptedTextGenerator(reply)).run(
        request(), "r"
    )
    assert result.is_fallback is True
    assert result.fallback_reason == "AI_ERROR"
    assert result.model_version == "template@1.0.0"


def test_llm_failure_or_bad_json_falls_back() -> None:
    for reply in (
        TextGenerationError("RateLimitError"),
        "no es json",
        "[1]",
        llm_json(sections=[]),
        llm_json(sections=["x"]),
        llm_json(sections=[{"title": "", "text": "x"}]),
    ):
        result = SummaryPipeline(settings(), ScriptedTextGenerator(reply)).run(
            request(), "r"
        )
        assert result.is_fallback is True


def test_code_fenced_json_is_accepted() -> None:
    reply = (
        "```json\n"
        + llm_json(sections=[{"title": "R", "text": "GMI 7,5 %."}])
        + "\n```"
    )
    result = SummaryPipeline(settings(), ScriptedTextGenerator(reply)).run(
        request(), "r"
    )
    assert result.is_fallback is False
