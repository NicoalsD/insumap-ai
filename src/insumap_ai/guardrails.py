"""Guardrail chain of the assistant (Modulo-IA 7): before and after the LLM.

Detectors are versioned pattern lists, never the LLM itself. Order: redaction,
dose and diagnosis, emergency symptoms, off topic and prompt injection. An
emergency wins over any other block.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Self, override

from insumap_ai.contract import EscalationLevel
from insumap_ai.redaction import PhiRedactor


class Guardrail(StrEnum):
    PHI_REDACTION = "PHI_REDACTION"
    DOSE_AND_DIAGNOSIS = "DOSE_AND_DIAGNOSIS"
    EMERGENCY_SYMPTOM = "EMERGENCY_SYMPTOM"
    OFF_TOPIC = "OFF_TOPIC"


DOSE_REPLY: Final = (
    "No puedo indicar dosis, cambios de medicamentos ni diagnósticos. "
    "Habla con tu médico o con tu equipo de atención."
)
EMERGENCY_REPLY: Final = (
    "Lo que describes puede ser una urgencia. Pide ayuda ahora: acude a urgencias "
    "o llama al número de emergencias. Si puedes, mide tu glucosa y no te quedes solo."
)
EMERGENCY_MESSAGE: Final = (
    "Acude a urgencias o llama al número de emergencias. Avisaremos a tu equipo "
    "de atención."
)
CONTACT_DOCTOR_MESSAGE: Final = (
    "Avisaremos a tu equipo de atención. Si empeoras, acude a urgencias."
)
OFF_TOPIC_REPLY: Final = (
    "Solo puedo ayudarte con educación sobre la diabetes y con el uso de Insumap."
)
FALLBACK_REPLY: Final = (
    "En este momento no puedo responder. Revisa el material de tu clínica o "
    "escríbele a tu equipo de atención."
)


def fold(text: str) -> str:
    """Lowercase without accents, so patterns match "dósis" and "DOSIS"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _patterns(*sources: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(source) for source in sources)


DOSE_REQUEST: Final = _patterns(
    r"\bcuant[ao]s? (unidades|ui\b|insulina)",
    r"\b(que|cual|cuanta) dosis\b",
    r"\b(subo|bajo|aumento|reduzco|cambio|ajusto|duplico|suspendo|dejo)\b.{0,20}"
    r"\b(dosis|insulina|unidades|metformina|pastilla|medicamento)",
    r"\bme (pongo|inyecto|aplico)\b.{0,15}\d",
    r"\b(debo|puedo) (dejar de )?(tomar|ponerme|inyectarme)\b",
    r"\b(tengo|sera que tengo|me diagnosticas?|es un|es una)\b.{0,30}"
    r"\b(cetoacidosis|neuropatia|retinopatia|nefropatia|infeccion|infarto|"
    r"gastroparesia|diagnostico)",
    r"\bque enfermedad tengo\b",
)
DOSE_OUTPUT: Final = _patterns(
    r"\b\d+([.,]\d+)?\s*(u|ui|unidades)\b",
    r"\b(aumenta|reduce|sube|baja|ajusta|suspende|duplica|disminuye|incrementa)\w*\b"
    r".{0,25}\b(dosis|insulina|basal|bolo|medicamento|metformina)",
    r"\b(deja|dejar) de (tomar|ponerte|inyectarte|usar)\b",
    r"\b(tienes|padeces|sufres de|usted tiene)\b.{0,20}\b(cetoacidosis|neuropatia|"
    r"retinopatia|nefropatia|infeccion|diabetes tipo)",
    r"\bte diagnostico\b",
)
EMERGENCY: Final = _patterns(
    r"\bconfundid[oa]|\bconfusion\b|\bdesorientad",
    r"\bdesmay|\bperd(i|io|imos) el conocimiento|\binconsciente",
    r"\bno (despierta|reacciona)",
    r"\bconvulsi",
    r"\b(no puedo|me cuesta|dificultad para) respirar|\bme falta el aire|\bme ahogo",
    r"\bvomit.{0,40}cetona|\bcetona.{0,40}vomit",
    r"\bdolor (fuerte )?(en el |de )?pecho",
    r"\bno puedo tragar",
    r"\b(glucosa|azucar) (muy|super) baja\b.{0,60}\b(tembl|sudor|mare|confus|desmay)",
)
CONTACT_DOCTOR: Final = _patterns(
    r"\btembl",
    r"\bsudor fri",
    r"\bmareado|\bmareo",
    r"\bvision borrosa",
    r"\bmucha sed\b",
    r"\borino mucho\b",
    r"\bcetonas?\b",
    r"\bhipoglucemia",
)
INJECTION: Final = _patterns(
    r"\b(ignora|olvida|omite)\b.{0,30}\b(instrucciones|reglas|indicaciones|prompt)",
    r"\b(actua|comportate|haz de cuenta) como\b",
    r"\b(eres ahora|a partir de ahora eres)\b",
    r"\bsystem prompt|\bprompt del sistema|\bjailbreak|\bmodo desarrollador",
    r"\bignore (all |the |your )?(previous )?instructions",
)


@dataclass
class GuardContext:
    """What one pass of the chain carries and decides."""

    text: str
    triggered: list[Guardrail] = field(default_factory=list)
    escalation: EscalationLevel = EscalationLevel.NONE
    blocked_reply: str | None = None

    def trigger(self, guardrail: Guardrail) -> None:
        if guardrail not in self.triggered:
            self.triggered.append(guardrail)

    def block(self, reply: str, *, priority: bool = False) -> None:
        if priority or self.blocked_reply is None:
            self.blocked_reply = reply


class GuardrailHandler(ABC):
    """One link of the chain: continue, modify the text or block.

    Pattern: P12 Chain of Responsibility
    """

    def __init__(self) -> None:
        self._next: GuardrailHandler | None = None

    def then(self, handler: GuardrailHandler) -> GuardrailHandler:
        self._next = handler
        return handler

    def handle(self, context: GuardContext) -> GuardContext:
        self.apply(context)
        if self._next is not None:
            return self._next.handle(context)
        return context

    @abstractmethod
    def apply(self, context: GuardContext) -> None: ...

    @classmethod
    def chain(cls, *handlers: GuardrailHandler) -> GuardrailHandler:
        first = handlers[0]
        current = first
        for handler in handlers[1:]:
            current = current.then(handler)
        return first


class PhiRedactionHandler(GuardrailHandler):
    def __init__(self, redactor: PhiRedactor) -> None:
        super().__init__()
        self._redactor = redactor

    @override
    def apply(self, context: GuardContext) -> None:
        redaction = self._redactor.redact(context.text)
        if redaction.changed:
            context.text = redaction.text
            context.trigger(Guardrail.PHI_REDACTION)


class DoseAndDiagnosisHandler(GuardrailHandler):
    """Blocks requests for doses or diagnoses (input) and any reply that gives
    them (output)."""

    def __init__(self, patterns: tuple[re.Pattern[str], ...]) -> None:
        super().__init__()
        self._patterns = patterns

    @classmethod
    def for_input(cls) -> Self:
        return cls(DOSE_REQUEST)

    @classmethod
    def for_output(cls) -> Self:
        return cls(DOSE_OUTPUT)

    @override
    def apply(self, context: GuardContext) -> None:
        folded = fold(context.text)
        if any(pattern.search(folded) for pattern in self._patterns):
            context.trigger(Guardrail.DOSE_AND_DIAGNOSIS)
            context.block(DOSE_REPLY)


class EmergencySymptomHandler(GuardrailHandler):
    """Alarm symptoms escalate to emergency and win over any other block; milder
    hypoglycemia symptoms ask the care team to look."""

    @override
    def apply(self, context: GuardContext) -> None:
        folded = fold(context.text)
        if any(pattern.search(folded) for pattern in EMERGENCY):
            context.trigger(Guardrail.EMERGENCY_SYMPTOM)
            context.escalation = EscalationLevel.EMERGENCY
            context.block(EMERGENCY_REPLY, priority=True)
        elif any(pattern.search(folded) for pattern in CONTACT_DOCTOR):
            context.escalation = EscalationLevel.CONTACT_DOCTOR


class OffTopicHandler(GuardrailHandler):
    """Prompt injection is off topic by definition. Other off-topic messages are
    flagged by the reply (``on_topic: false``) and handled by the pipeline."""

    @override
    def apply(self, context: GuardContext) -> None:
        if context.escalation is EscalationLevel.EMERGENCY:
            return
        folded = fold(context.text)
        if any(pattern.search(folded) for pattern in INJECTION):
            context.trigger(Guardrail.OFF_TOPIC)
            context.block(OFF_TOPIC_REPLY)


def input_chain(redactor: PhiRedactor) -> GuardrailHandler:
    return GuardrailHandler.chain(
        PhiRedactionHandler(redactor),
        DoseAndDiagnosisHandler.for_input(),
        EmergencySymptomHandler(),
        OffTopicHandler(),
    )


def output_chain(redactor: PhiRedactor) -> GuardrailHandler:
    return GuardrailHandler.chain(
        PhiRedactionHandler(redactor),
        DoseAndDiagnosisHandler.for_output(),
    )
