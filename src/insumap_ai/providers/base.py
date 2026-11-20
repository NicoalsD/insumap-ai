"""The three model interfaces the rest of the service knows (AGENTS.md 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class TextGenerationError(Exception):
    """The LLM failed, timed out, refused or answered something unusable."""


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class Prompt:
    """A system instruction plus the conversation, built by ``PromptBuilder``."""

    system: str
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    version: str


class TextGenerator(ABC):
    """Generates text from a prompt. Implementations never log the prompt."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...

    @abstractmethod
    def generate(self, prompt: Prompt, *, json_output: bool) -> str:
        """Returns the text, or raises ``TextGenerationError``."""


@dataclass(frozen=True, slots=True)
class GlucoseSeries:
    timestamps: tuple[datetime, ...]
    values: tuple[float, ...]
    interval_minutes: int


@dataclass(frozen=True, slots=True)
class QuantilePoint:
    horizon_minutes: int
    p10: float
    p50: float
    p90: float
    sigma: float
    """Spread used for the threshold probabilities (mg/dL)."""


@dataclass(frozen=True, slots=True)
class ForecastResult:
    slope_mg_dl_min: float
    points: tuple[QuantilePoint, ...]


class Forecaster(ABC):
    """Quantile forecast of a CGM series at the given horizons."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...

    @abstractmethod
    def forecast(
        self, series: GlucoseSeries, horizons_minutes: tuple[int, ...]
    ) -> ForecastResult: ...


@dataclass(frozen=True, slots=True)
class RiskContribution:
    name: str
    value: float | int | None
    unit: str | None
    weight: float
    """Signed contribution to the logit; positive increases the risk."""


@dataclass(frozen=True, slots=True)
class RiskScore:
    probability: float
    contributions: tuple[RiskContribution, ...] = field(default_factory=tuple)


class RiskModel(ABC):
    """Scores the deterioration (IA2) or dropout (IA7) risk from features."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...

    @property
    @abstractmethod
    def calibrated(self) -> bool: ...

    @abstractmethod
    def deterioration(self, features: dict[str, float | None]) -> RiskScore: ...

    @abstractmethod
    def dropout(self, features: dict[str, float | None]) -> RiskScore: ...
