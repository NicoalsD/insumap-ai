"""Removes personal data from free text before it can reach the LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# Order matters: e-mails before documents, record numbers before bare digits.
_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("[CORREO]", re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")),
    (
        "[HISTORIA]",
        re.compile(
            r"\b(historia(\s+cl[ií]nica)?|HC|MRN|expediente)\s*(n[uú]mero|no\.?|#|:)?"
            r"\s*[A-Z0-9-]{4,}",
            re.IGNORECASE,
        ),
    ),
    (
        "[DOCUMENTO]",
        re.compile(
            r"\b(c[eé]dula|CC|C\.C\.|documento|pasaporte|DNI|TI|NIT)\s*(es\s+|n[uú]mero|no\.?|#|:)?"
            r"\s*[\d.\s-]{5,15}\d",
            re.IGNORECASE,
        ),
    ),
    (
        "[TELEFONO]",
        re.compile(
            r"(?<![\d.,])(\+\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}(?![\d.,])"
        ),
    ),
    ("[DOCUMENTO]", re.compile(r"(?<![\d.,])\d{7,12}(?![\d.,])")),
    ("[DOCUMENTO]", re.compile(r"(?<![\d.,])\d{1,3}(\.\d{3}){2,}(?![\d,])")),
    (
        "[DIRECCION]",
        re.compile(
            r"\b(calle|carrera|cra\.?|cl\.?|avenida|av\.?|diagonal|transversal|"
            r"autopista)\s+\d+[a-z]?\s*(bis)?\s*(#|n[uú]mero|no\.?)\s*\d+[a-z]?"
            r"(\s*-\s*\d+)?",
            re.IGNORECASE,
        ),
    ),
    (
        r"\1[NOMBRE]",
        re.compile(
            r"\b((?i:me llamo|mi nombre es|soy|nombre:)\s+)"
            r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Redaction:
    text: str
    changed: bool


class PhiRedactor:
    """Replaces names, documents, record numbers, phones, e-mails and addresses
    with placeholders such as ``[NOMBRE]``."""

    def redact(self, text: str) -> Redaction:
        result = text
        for placeholder, pattern in _RULES:
            result = pattern.sub(placeholder, result)
        return Redaction(text=result, changed=result != text)
