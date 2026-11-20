"""Service settings, read only from environment variables (Contrato-IA 10.2).

Secrets have no default: without ``AI_SERVICE_TOKEN`` the service does not start.
Clinical parameters never live here; they travel in each request.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_TOKEN_BYTES: Final = 32


class ProviderFamily(StrEnum):
    """Which ``AiProviderFactory`` builds the models."""

    OPENAI_COMPATIBLE = "openai_compatible"
    OFFLINE = "offline"


class Settings(BaseSettings):
    """Every knob of the service. Technical limits only, never clinical targets."""

    model_config = SettingsConfigDict(
        extra="ignore", frozen=True, env_ignore_empty=True
    )

    ai_service_token: SecretStr
    ai_service_token_previous: SecretStr | None = None
    ai_provider_family: ProviderFamily = ProviderFamily.OFFLINE

    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=15.0, gt=0)
    llm_max_output_tokens: int = Field(default=2048, ge=64)
    llm_reasoning_effort: str | None = None
    """Optional ``reasoning_effort`` for thinking models (``low``, ``none``...)."""

    max_body_bytes: int = Field(default=512 * 1024, ge=1024)
    max_forecast_points: int = Field(default=2048, ge=1)
    forecast_min_points: int = Field(default=36, ge=2)
    forecast_slope_window_minutes: int = Field(default=30, ge=5)
    forecast_residual_window_points: int = Field(default=36, ge=4)
    forecast_spread_floor_mg_dl: float = Field(default=4.0, ge=0)
    forecast_trend_threshold_mg_dl_min: float = Field(default=1.0, gt=0)
    forecast_damping_per_hour: float = Field(default=0.5, gt=0, le=1)
    max_doses: int = Field(default=500, ge=1)
    max_context_docs: int = Field(default=8, ge=1)
    max_context_doc_chars: int = Field(default=4000, ge=100)
    max_message_chars: int = Field(default=1000, ge=1)
    max_history_turns: int = Field(default=10, ge=0)
    glucose_plausible_min: float = Field(default=20.0, gt=0)
    glucose_plausible_max: float = Field(default=600.0, gt=0)
    risk_min_data_sufficiency_pct: float = Field(default=70.0, ge=0, le=100)
    risk_tier_medium: float = Field(default=0.2, gt=0, lt=1)
    risk_tier_high: float = Field(default=0.5, gt=0, lt=1)
    risk_tier_critical: float = Field(default=0.8, gt=0, lt=1)

    api_docs_enabled: bool = True

    @field_validator("ai_service_token", "ai_service_token_previous")
    @classmethod
    def token_is_long_enough(cls, value: SecretStr | None) -> SecretStr | None:
        if (
            value is not None
            and len(value.get_secret_value().encode()) < MIN_TOKEN_BYTES
        ):
            raise ValueError("AI_SERVICE_TOKEN must have at least 32 bytes")
        return value

    @model_validator(mode="after")
    def llm_is_complete(self) -> Self:
        if self.ai_provider_family is ProviderFamily.OPENAI_COMPATIBLE and not (
            self.llm_base_url and self.llm_model and self.llm_api_key
        ):
            raise ValueError(
                "openai_compatible needs LLM_BASE_URL, LLM_MODEL and LLM_API_KEY"
            )
        if not (self.risk_tier_medium < self.risk_tier_high < self.risk_tier_critical):
            raise ValueError("risk tier cutoffs must increase")
        if self.glucose_plausible_min >= self.glucose_plausible_max:
            raise ValueError("GLUCOSE_PLAUSIBLE_MIN must be below the maximum")
        return self
