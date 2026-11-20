"""IA6 pipeline: guardrails around the LLM, citations and escalation."""

from __future__ import annotations

from insumap_ai.contract import AssistantReplyRequest
from insumap_ai.guardrails import DOSE_REPLY, FALLBACK_REPLY, OFF_TOPIC_REPLY
from insumap_ai.pipeline import AssistantPipeline
from insumap_ai.providers.base import ChatRole, TextGenerationError
from tests.conftest import ScriptedTextGenerator, assistant_body, llm_json, settings


def ask(
    message: str,
    *replies: str | Exception,
    docs: bool = True,
    history: list[dict[str, str]] | None = None,
):  # type: ignore[no-untyped-def]
    body = assistant_body(message, docs)
    if history:
        body["history"] = history
    llm = ScriptedTextGenerator(*replies)
    result = AssistantPipeline(settings(), llm).run(
        AssistantReplyRequest.model_validate(body), "rid"
    )
    return result, llm


def test_answers_with_the_material_and_cites_it() -> None:
    result, llm = ask(
        "¿Qué hago si me siento tembloroso y con sudor frío?",
        llm_json(
            reply="Mide tu glucosa ahora y avisa a tu equipo.",
            citations=["doc-hipo-01", "doc-falso"],
            on_topic=True,
        ),
    )
    assert result.is_fallback is False
    assert result.reply_text == "Mide tu glucosa ahora y avisa a tu equipo."
    assert result.citations == ["doc-hipo-01"]
    assert result.escalation.level == "CONTACT_DOCTOR"
    assert "Qué es la hipoglucemia" in result.explanation.summary
    assert "doc-hipo-01" in llm.prompts[0].system


def test_history_keeps_roles() -> None:
    _, llm = ask(
        "¿Y después?",
        llm_json(reply="Vuelve a medir en 15 minutos.", citations=[]),
        history=[
            {"role": "PATIENT", "text": "Tengo la glucosa baja"},
            {"role": "ASSISTANT", "text": "Come algo dulce."},
        ],
    )
    roles = [m.role for m in llm.prompts[0].messages]
    assert roles == [ChatRole.USER, ChatRole.ASSISTANT, ChatRole.USER]


def test_dose_request_never_reaches_the_llm() -> None:
    result, llm = ask("¿Cuántas unidades me pongo?")
    assert result.reply_text == DOSE_REPLY
    assert result.is_fallback is True
    assert llm.prompts == []


def test_reply_with_a_dose_is_replaced() -> None:
    result, _ = ask(
        "¿Cómo corrijo una glucosa alta?",
        llm_json(reply="Ponte 2 unidades extra.", citations=[]),
    )
    assert result.reply_text == DOSE_REPLY
    assert "DOSE_AND_DIAGNOSIS" in result.guardrails_triggered


def test_off_topic_flag_from_the_reply() -> None:
    result, _ = ask(
        "¿Quién ganó el partido?", llm_json(reply="No sé de fútbol.", on_topic=False)
    )
    assert result.reply_text == OFF_TOPIC_REPLY
    assert result.guardrails_triggered == ["OFF_TOPIC"]


def test_llm_failure_answers_the_predefined_text() -> None:
    for reply in (
        TextGenerationError("RateLimitError"),
        "{}",
        "texto",
        llm_json(reply="   "),
    ):
        result, _ = ask("¿Qué es la HbA1c?", reply)
        assert result.reply_text == FALLBACK_REPLY
        assert result.is_fallback is True


def test_personal_data_in_the_message_is_redacted_before_the_llm() -> None:
    _, llm = ask(
        "Me llamo Ana Pérez, ¿qué es la HbA1c?",
        llm_json(reply="Es un promedio de tu glucosa.", citations=[]),
    )
    assert "Ana Pérez" not in llm.prompts[0].messages[-1].content


def test_long_reply_is_cut_at_a_sentence() -> None:
    long_reply = "Frase corta. " * 200
    result, _ = ask("Explícame la diabetes", llm_json(reply=long_reply, citations=[]))
    assert len(result.reply_text) <= 1000
    assert result.reply_text.endswith(".")
