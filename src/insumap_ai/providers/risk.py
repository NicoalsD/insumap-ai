"""Logistic risk scorer (IA2 and IA7) driven by its versioned model card."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from typing import Any, Final, override

from insumap_ai.providers.base import ModelInfo, RiskContribution, RiskModel, RiskScore

CARD_FILE: Final = "risk-logistic.json"
TERM_MIN: Final = -2.0
TERM_MAX: Final = 3.0
"""A single feature moves the logit at most this many weights either way."""


@dataclass(frozen=True, slots=True)
class Term:
    feature: str
    unit: str | None
    reference: float
    scale: float
    weight: float


@dataclass(frozen=True, slots=True)
class LogisticCard:
    intercept: float
    terms: tuple[Term, ...]


def load_card(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """The bundled model card, or ``raw`` for tests."""
    if raw is not None:
        return raw
    text = resources.files("insumap_ai.model_cards").joinpath(CARD_FILE).read_text()
    card: dict[str, Any] = json.loads(text)
    return card


def _logistic(section: dict[str, Any]) -> LogisticCard:
    return LogisticCard(
        intercept=float(section["intercept"]),
        terms=tuple(Term(**term) for term in section["terms"]),
    )


class LogisticRiskModel(RiskModel):
    """``sigmoid(intercept + sum(weight * clamp((x - reference) / scale)))``.

    Missing features add nothing, so a sparse request leans to the intercept.
    """

    def __init__(self, card: dict[str, Any]) -> None:
        self._info = ModelInfo(name=str(card["name"]), version=str(card["version"]))
        self._calibrated = bool(card["calibrated"])
        self._deterioration = _logistic(card["deterioration"])
        self._dropout = _logistic(card["dropout"])

    @property
    @override
    def info(self) -> ModelInfo:
        return self._info

    @property
    @override
    def calibrated(self) -> bool:
        return self._calibrated

    @override
    def deterioration(self, features: dict[str, float | None]) -> RiskScore:
        return _score(self._deterioration, features)

    @override
    def dropout(self, features: dict[str, float | None]) -> RiskScore:
        return _score(self._dropout, features)


def _score(card: LogisticCard, features: dict[str, float | None]) -> RiskScore:
    logit = card.intercept
    contributions = []
    for term in card.terms:
        value = features.get(term.feature)
        if value is None:
            continue
        z = min(TERM_MAX, max(TERM_MIN, (value - term.reference) / term.scale))
        weight = term.weight * z
        logit += weight
        contributions.append(
            RiskContribution(
                name=term.feature, value=value, unit=term.unit, weight=weight
            )
        )
    probability = 1 / (1 + math.exp(-logit))
    return RiskScore(probability=probability, contributions=tuple(contributions))
