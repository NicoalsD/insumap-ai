"""P09 PhiRedactingTextGenerator: no personal data reaches the generator."""

from __future__ import annotations

import pytest

from insumap_ai.providers.base import ChatMessage, ChatRole, Prompt
from insumap_ai.providers.text import PhiRedactingTextGenerator
from insumap_ai.redaction import PhiRedactor
from tests.conftest import ScriptedTextGenerator


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("me llamo Ana María Pérez", "me llamo [NOMBRE]"),
        ("mi cédula es 1.020.345.678", "mi [DOCUMENTO]"),
        ("documento 52345678", "[DOCUMENTO]"),
        ("escríbeme a ana.perez@correo.com", "escríbeme a [CORREO]"),
        ("mi celular 300 123 4567", "mi celular [TELEFONO]"),
        ("vivo en Calle 45 # 12-30", "vivo en [DIRECCION]"),
        ("HC: AB12345", "[HISTORIA]"),
    ],
)
def test_redacts_personal_data(text: str, expected: str) -> None:
    assert PhiRedactor().redact(text).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "TIR 48,2 % y glucosa media 176 mg/dL",
        "soy diabética tipo 1",
        "lecturas de 120, 130 y 140",
        "GMI 7.5 %",
    ],
)
def test_keeps_clinical_figures(text: str) -> None:
    redaction = PhiRedactor().redact(text)
    assert redaction.text == text and not redaction.changed


def test_no_personal_data_reaches_the_generator() -> None:
    inner = ScriptedTextGenerator("ok")
    generator = PhiRedactingTextGenerator(inner, PhiRedactor())
    prompt = Prompt(
        system="Paciente Ana: correo ana@x.co",
        messages=(ChatMessage(ChatRole.USER, "me llamo Ana Pérez, cel 300 123 4567"),),
    )
    assert generator.generate(prompt, json_output=False) == "ok"
    sent = inner.prompts[0]
    assert "ana@x.co" not in sent.system
    assert "Ana Pérez" not in sent.messages[0].content
    assert "300 123 4567" not in sent.messages[0].content
    assert generator.info == inner.info
