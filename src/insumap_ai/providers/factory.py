"""Families of providers chosen by ``AI_PROVIDER_FAMILY``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from openai import OpenAI

from insumap_ai.config import ProviderFamily, Settings
from insumap_ai.providers.base import Forecaster, RiskModel, TextGenerator
from insumap_ai.providers.forecasting import (
    DampedTrendForecasterCreator,
    ForecastOptions,
    LinearForecasterCreator,
)
from insumap_ai.providers.risk import LogisticRiskModel, load_card
from insumap_ai.providers.text import (
    OfflineTextGenerator,
    OpenAiCompatibleTextGeneratorAdapter,
    PhiRedactingTextGenerator,
)
from insumap_ai.redaction import PhiRedactor


class AiProviderFactory(ABC):
    """Creates one consistent family of models; the rest of the service only knows
    ``TextGenerator``, ``Forecaster`` and ``RiskModel``.

    Pattern: P03 Abstract Factory
    """

    family: ProviderFamily

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abstractmethod
    def create_text_generator(self) -> TextGenerator: ...

    def create_forecaster(self) -> Forecaster:
        return DampedTrendForecasterCreator(self._forecast_options()).create()

    def create_fallback_forecaster(self) -> Forecaster:
        return LinearForecasterCreator(self._forecast_options()).create()

    def create_risk_model(self) -> RiskModel:
        return LogisticRiskModel(load_card())

    def _forecast_options(self) -> ForecastOptions:
        s = self._settings
        return ForecastOptions(
            slope_window_minutes=s.forecast_slope_window_minutes,
            residual_window_points=s.forecast_residual_window_points,
            damping_per_hour=s.forecast_damping_per_hour,
            spread_floor=s.forecast_spread_floor_mg_dl,
            plausible_min=s.glucose_plausible_min,
            plausible_max=s.glucose_plausible_max,
        )


class OpenAiCompatibleProviderFactory(AiProviderFactory):
    """The remote LLM behind the redacting decorator."""

    family = ProviderFamily.OPENAI_COMPATIBLE

    @override
    def create_text_generator(self) -> TextGenerator:
        s = self._settings
        if not (s.llm_base_url and s.llm_model and s.llm_api_key):
            raise ValueError("the LLM is not configured")
        client = OpenAI(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key.get_secret_value(),
            timeout=s.llm_timeout_seconds,
            max_retries=0,
        )
        adapter = OpenAiCompatibleTextGeneratorAdapter(
            client=client,
            model=s.llm_model,
            max_output_tokens=s.llm_max_output_tokens,
            reasoning_effort=s.llm_reasoning_effort,
        )
        return PhiRedactingTextGenerator(adapter, PhiRedactor())


class OfflineProviderFactory(AiProviderFactory):
    """Deterministic family for tests and CI: no network, no weights."""

    family = ProviderFamily.OFFLINE

    @override
    def create_text_generator(self) -> TextGenerator:
        return PhiRedactingTextGenerator(OfflineTextGenerator(), PhiRedactor())


def provider_factory(settings: Settings) -> AiProviderFactory:
    if settings.ai_provider_family is ProviderFamily.OPENAI_COMPATIBLE:
        return OpenAiCompatibleProviderFactory(settings)
    return OfflineProviderFactory(settings)
