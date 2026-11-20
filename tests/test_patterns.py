"""One test per design pattern not covered elsewhere: P01, P02, P03, P04, P06,
P09 (CachingForecaster), P14 and P20."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import httpx
import pytest
from openai import OpenAI

from insumap_ai.config import ProviderFamily
from insumap_ai.prompts import PromptBuilder
from insumap_ai.providers.base import (
    ChatMessage,
    ChatRole,
    Forecaster,
    ForecastResult,
    GlucoseSeries,
    ModelInfo,
    Prompt,
    TextGenerationError,
)
from insumap_ai.providers.factory import (
    OfflineProviderFactory,
    OpenAiCompatibleProviderFactory,
    provider_factory,
)
from insumap_ai.providers.forecasting import (
    CachingForecaster,
    DampedTrendForecaster,
    DampedTrendForecasterCreator,
    ForecastOptions,
    LinearForecasterCreator,
)
from insumap_ai.providers.text import (
    OfflineTextGenerator,
    OpenAiCompatibleTextGeneratorAdapter,
    PhiRedactingTextGenerator,
)
from insumap_ai.registry import ModelRegistry
from insumap_ai.windows import SlidingWindowIterator
from tests.conftest import settings

OPTIONS = ForecastOptions(
    slope_window_minutes=30,
    residual_window_points=36,
    damping_per_hour=0.5,
    spread_floor=4.0,
    plausible_min=20.0,
    plausible_max=600.0,
)


def series(values: list[float]) -> GlucoseSeries:
    start = datetime.fromisoformat("2026-11-14T20:00:00-05:00")
    return GlucoseSeries(
        timestamps=tuple(start + timedelta(minutes=5 * i) for i in range(len(values))),
        values=tuple(values),
        interval_minutes=5,
    )


def test_p01_registry_is_a_singleton() -> None:
    ModelRegistry.reset()
    assert ModelRegistry.instance() is ModelRegistry.instance()
    registry = ModelRegistry.instance()
    assert not registry.ready
    with pytest.raises(RuntimeError):
        _ = registry.models
    loaded = registry.load(settings())
    assert registry.load(settings()) is loaded and registry.ready
    ModelRegistry.reset()
    assert ModelRegistry.instance() is not registry


def test_p02_creators_build_cached_forecasters() -> None:
    damped = DampedTrendForecasterCreator(OPTIONS).create()
    linear = LinearForecasterCreator(OPTIONS).create()
    assert isinstance(damped, CachingForecaster) and isinstance(
        linear, CachingForecaster
    )
    assert damped.info.name == "damped-trend"
    assert linear.info.version == "fallback-linear@1.0.0"


def test_p03_factory_families() -> None:
    offline = provider_factory(settings())
    assert isinstance(offline, OfflineProviderFactory)
    assert offline.family is ProviderFamily.OFFLINE
    assert offline.create_text_generator().info.name == "offline-template"
    remote = provider_factory(
        settings(
            ai_provider_family="openai_compatible",
            llm_base_url="https://llm.test/v1",
            llm_model="gemini-test",
            llm_api_key="k",
        )
    )
    assert isinstance(remote, OpenAiCompatibleProviderFactory)
    generator = remote.create_text_generator()
    assert isinstance(generator, PhiRedactingTextGenerator)
    assert generator.info.name == "gemini-test"
    assert remote.create_risk_model().info.name == "risk-logistic"


def test_p04_builder() -> None:
    prompt = (
        PromptBuilder()
        .system("S")
        .document("d1", "T", "texto")
        .turn(ChatRole.USER, "hola")
        .build()
    )
    assert prompt.system == "S\n\n## Material aprobado [d1] T\ntexto"
    assert prompt.messages == (ChatMessage(ChatRole.USER, "hola"),)
    with pytest.raises(ValueError, match="system"):
        PromptBuilder().turn(ChatRole.USER, "x").build()
    with pytest.raises(ValueError, match="user turn"):
        PromptBuilder().system("S").build()


def adapter(handler) -> OpenAiCompatibleTextGeneratorAdapter:  # type: ignore[no-untyped-def]
    client = OpenAI(
        base_url="https://llm.test/v1",
        api_key="k",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return OpenAiCompatibleTextGeneratorAdapter(
        client=client,
        model="gemini-test",
        max_output_tokens=256,
        reasoning_effort="low",
    )


def completion(content: str | None, finish: str = "stop") -> dict[str, object]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


PROMPT = Prompt(system="S", messages=(ChatMessage(ChatRole.USER, "hola"),))


def test_p06_adapter_sends_the_openai_request() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["path"] = request.url.path
        return httpx.Response(200, json=completion('{"ok": true}'))

    assert adapter(handler).generate(PROMPT, json_output=True) == '{"ok": true}'
    assert seen["path"] == "/v1/chat/completions"
    assert seen["temperature"] == 0
    assert seen["model"] == "gemini-test"
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["reasoning_effort"] == "low"
    assert seen["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hola"},
    ]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, json={"error": {"message": "quota"}}),
        httpx.Response(500, json={}),
        httpx.Response(200, json=completion(None)),
        httpx.Response(200, json=completion("cortado", finish="length")),
        httpx.Response(200, json={**completion("x"), "choices": []}),
    ],
)
def test_p06_adapter_turns_failures_into_text_generation_errors(
    response: httpx.Response,
) -> None:
    with pytest.raises(TextGenerationError):
        adapter(lambda _: response).generate(PROMPT, json_output=False)


def test_offline_generator_always_declines() -> None:
    with pytest.raises(TextGenerationError):
        OfflineTextGenerator().generate(PROMPT, json_output=True)


def test_p09_caching_forecaster_calls_the_model_once() -> None:
    calls: list[int] = []

    class Counting(Forecaster):
        @property
        def info(self) -> ModelInfo:
            return ModelInfo("c", "1")

        def forecast(self, s: GlucoseSeries, h: tuple[int, ...]) -> ForecastResult:
            calls.append(1)
            return ForecastResult(slope_mg_dl_min=0.0, points=())

    cached = CachingForecaster(Counting(), capacity=1)
    data = series([100.0, 101.0])
    cached.forecast(data, (30,))
    cached.forecast(data, (30,))
    assert len(calls) == 1
    cached.forecast(series([90.0, 91.0]), (30,))
    cached.forecast(data, (30,))
    assert len(calls) == 3
    assert cached.info.name == "c"


def test_p14_sliding_window() -> None:
    assert list(SlidingWindowIterator([1, 2, 3, 4], 2)) == [(1, 2), (2, 3), (3, 4)]
    assert list(SlidingWindowIterator([1, 2, 3, 4, 5], 2, step=2)) == [(1, 2), (3, 4)]
    assert list(SlidingWindowIterator([1], 2)) == []
    with pytest.raises(ValueError, match="positive"):
        SlidingWindowIterator([1], 0)


def test_damped_trend_quantiles_and_damping() -> None:
    model = DampedTrendForecaster(
        slope_window_minutes=30,
        residual_window_points=36,
        damping_per_hour=0.5,
        spread_floor=4.0,
        plausible_min=20.0,
        plausible_max=600.0,
    )
    result = model.forecast(series([200 - 5 * i for i in range(40)]), (30, 60, 120))
    assert result.slope_mg_dl_min == pytest.approx(-1.0)
    drops = [5.0 - p.p50 for p in result.points]  # last value is 5
    assert all(p.p10 <= p.p50 <= p.p90 for p in result.points)
    # Clipped at the plausible minimum instead of going negative.
    assert all(p.p10 >= 20.0 for p in result.points)
    assert drops


def test_damping_slows_long_horizons() -> None:
    model = DampedTrendForecaster(
        slope_window_minutes=30,
        residual_window_points=36,
        damping_per_hour=0.5,
        spread_floor=4.0,
        plausible_min=20.0,
        plausible_max=600.0,
    )
    result = model.forecast(series([300 - 1 * i for i in range(40)]), (60, 120))
    first, second = (261.0 - p.p50 for p in result.points)
    assert second < 2 * first
