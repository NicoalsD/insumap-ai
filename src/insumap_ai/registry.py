"""The models of the process, built once (Contrato-IA 9.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import ClassVar, Self

from insumap_ai.config import Settings
from insumap_ai.contract import ModelCard, ModelStatus
from insumap_ai.providers.base import Forecaster, RiskModel, TextGenerator
from insumap_ai.providers.factory import provider_factory


@dataclass(frozen=True, slots=True)
class LoadedModels:
    family: str
    text: TextGenerator
    forecaster: Forecaster
    fallback_forecaster: Forecaster
    risk: RiskModel
    loaded_at: datetime


class ModelRegistry:
    """Loads the models of the configured family once per process.

    Pattern: P01 Singleton
    """

    _instance: ClassVar[ModelRegistry | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        self._models: LoadedModels | None = None
        self._failed = False

    @classmethod
    def instance(cls) -> Self:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset(cls) -> None:
        """Forget the instance (tests and settings reloads)."""
        with cls._lock:
            cls._instance = None

    def load(self, settings: Settings) -> LoadedModels:
        if self._models is None:
            try:
                factory = provider_factory(settings)
                self._models = LoadedModels(
                    family=factory.family.value,
                    text=factory.create_text_generator(),
                    forecaster=factory.create_forecaster(),
                    fallback_forecaster=factory.create_fallback_forecaster(),
                    risk=factory.create_risk_model(),
                    loaded_at=datetime.now(UTC),
                )
            except Exception:
                self._failed = True
                raise
        return self._models

    @property
    def ready(self) -> bool:
        return self._models is not None and not self._failed

    @property
    def models(self) -> LoadedModels:
        if self._models is None:
            raise RuntimeError("models are not loaded")
        return self._models

    def cards(self, settings: Settings) -> list[ModelCard]:
        m = self.models
        status = ModelStatus.READY
        cutoffs = {
            "medium": settings.risk_tier_medium,
            "high": settings.risk_tier_high,
            "critical": settings.risk_tier_critical,
        }
        return [
            ModelCard(
                name=m.forecaster.info.name,
                version=m.forecaster.info.version,
                capability="glucose-forecasts",
                status=status,
                loaded_at=m.loaded_at,
            ),
            ModelCard(
                name=m.risk.info.name,
                version=m.risk.info.version,
                capability="risk-assessments",
                status=status,
                loaded_at=m.loaded_at,
                risk_tier_cutoffs=cutoffs,
            ),
            ModelCard(
                name=m.text.info.name,
                version=m.text.info.version,
                capability="clinical-summaries",
                status=status,
                loaded_at=m.loaded_at,
            ),
            ModelCard(
                name=m.text.info.name,
                version=m.text.info.version,
                capability="assistant-replies",
                status=status,
                loaded_at=m.loaded_at,
            ),
        ]
