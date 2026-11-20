"""Deterministic glucose forecasters (IA1) and their creators.

``DampedTrendForecaster`` is the model of this release: a damped linear trend
whose spread comes from the recent one-step volatility. ``LinearForecaster`` is
the service's own fallback (Contrato-IA 8.1): the plain rate of change of the
last 30 minutes. Chronos-Bolt will join as another creator.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean, pstdev
from typing import Final, override

from insumap_ai.providers.base import (
    Forecaster,
    ForecastResult,
    GlucoseSeries,
    ModelInfo,
    QuantilePoint,
)
from insumap_ai.windows import SlidingWindowIterator

Z_90: Final = 1.2815515655446004
"""Standard normal quantile of 0.9: p10 and p90 are ``p50 -/+ Z_90 * sigma``."""
SECONDS_PER_MINUTE: Final = 60.0
MINUTES_PER_HOUR: Final = 60.0


def recent_slope(series: GlucoseSeries, window_minutes: int) -> float:
    """Least-squares slope (mg/dL per minute) over the last ``window_minutes``."""
    last = series.timestamps[-1]
    xs: list[float] = []
    ys: list[float] = []
    for moment, value in zip(series.timestamps, series.values, strict=True):
        minutes = (moment - last).total_seconds() / SECONDS_PER_MINUTE
        if minutes >= -window_minutes:
            xs.append(minutes)
            ys.append(value)
    if len(xs) < 2:  # noqa: PLR2004 - a slope needs two points
        return 0.0
    mean_x, mean_y = fmean(xs), fmean(ys)
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / den


def step_volatility(series: GlucoseSeries, window_points: int) -> float:
    """Standard deviation of the per-minute change between consecutive readings."""
    tail = list(zip(series.timestamps, series.values, strict=True))[-window_points:]
    rates: list[float] = []
    for (t0, v0), (t1, v1) in SlidingWindowIterator(tail, 2):
        minutes = (t1 - t0).total_seconds() / SECONDS_PER_MINUTE
        if minutes > 0:
            rates.append((v1 - v0) / minutes)
    if len(rates) < 2:  # noqa: PLR2004 - a deviation needs two rates
        return 0.0
    return pstdev(rates)


class DampedTrendForecaster(Forecaster):
    """Damped trend with a spread that grows with the horizon.

    The rate of change decays by ``damping_per_hour`` each hour, so long
    horizons do not run away; the spread is the recent volatility of the rate,
    accumulated as a random walk over the horizon, and never below ``floor``.
    """

    def __init__(
        self,
        *,
        slope_window_minutes: int,
        residual_window_points: int,
        damping_per_hour: float,
        spread_floor: float,
        plausible_min: float,
        plausible_max: float,
    ) -> None:
        self._slope_window = slope_window_minutes
        self._residual_window = residual_window_points
        self._phi = damping_per_hour ** (1 / MINUTES_PER_HOUR)
        self._floor = spread_floor
        self._min = plausible_min
        self._max = plausible_max

    @property
    @override
    def info(self) -> ModelInfo:
        return ModelInfo(name="damped-trend", version="0.1.0+damped-trend")

    @override
    def forecast(
        self, series: GlucoseSeries, horizons_minutes: tuple[int, ...]
    ) -> ForecastResult:
        slope = recent_slope(series, self._slope_window)
        volatility = step_volatility(series, self._residual_window)
        last = series.values[-1]
        points = []
        for horizon in horizons_minutes:
            travelled = self._damped_minutes(horizon)
            p50 = last + slope * travelled
            sigma = max(
                self._floor, volatility * math.sqrt(horizon) * travelled / horizon
            )
            points.append(self._point(horizon, p50, sigma))
        return ForecastResult(slope_mg_dl_min=slope, points=tuple(points))

    def _damped_minutes(self, horizon: int) -> float:
        if self._phi >= 1:
            return float(horizon)
        phi: float = self._phi
        return phi * (1 - phi**horizon) / (1 - phi)

    def _point(self, horizon: int, p50: float, sigma: float) -> QuantilePoint:
        return quantile_point(horizon, p50, sigma, self._min, self._max)


def quantile_point(
    horizon: int, p50: float, sigma: float, low: float, high: float
) -> QuantilePoint:
    """The 10/50/90 quantiles of a normal spread, kept inside the sensor range."""

    def clip(value: float) -> float:
        return min(high, max(low, value))

    center = clip(p50)
    return QuantilePoint(
        horizon_minutes=horizon,
        p10=round(clip(center - Z_90 * sigma), 1),
        p50=round(center, 1),
        p90=round(clip(center + Z_90 * sigma), 1),
        sigma=sigma,
    )


class LinearForecaster(Forecaster):
    """The fallback: last value plus the rate of the last 30 minutes, undamped."""

    def __init__(
        self,
        *,
        slope_window_minutes: int,
        spread_floor: float,
        plausible_min: float,
        plausible_max: float,
    ) -> None:
        self._inner = DampedTrendForecaster(
            slope_window_minutes=slope_window_minutes,
            residual_window_points=2,
            damping_per_hour=1.0,
            spread_floor=spread_floor,
            plausible_min=plausible_min,
            plausible_max=plausible_max,
        )
        self._floor = spread_floor
        self._min = plausible_min
        self._max = plausible_max

    @property
    @override
    def info(self) -> ModelInfo:
        return ModelInfo(name="fallback-linear", version="fallback-linear@1.0.0")

    @override
    def forecast(
        self, series: GlucoseSeries, horizons_minutes: tuple[int, ...]
    ) -> ForecastResult:
        result = self._inner.forecast(series, horizons_minutes)
        # Without a volatility estimate the spread widens with the square root
        # of the horizon in sensor steps.
        widened = tuple(
            quantile_point(
                point.horizon_minutes,
                point.p50,
                self._floor
                * math.sqrt(point.horizon_minutes / series.interval_minutes),
                self._min,
                self._max,
            )
            for point in result.points
        )
        return ForecastResult(slope_mg_dl_min=result.slope_mg_dl_min, points=widened)


class CachingForecaster(Forecaster):
    """Remembers the last results by input, invisible to the contract.

    Pattern: P09 Decorator
    """

    def __init__(self, inner: Forecaster, capacity: int = 256) -> None:
        self._inner = inner
        self._capacity = capacity
        self._cache: OrderedDict[str, ForecastResult] = OrderedDict()

    @property
    @override
    def info(self) -> ModelInfo:
        return self._inner.info

    @override
    def forecast(
        self, series: GlucoseSeries, horizons_minutes: tuple[int, ...]
    ) -> ForecastResult:
        key = _series_key(series, horizons_minutes)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self._inner.forecast(series, horizons_minutes)
        self._cache[key] = result
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return result


def _series_key(series: GlucoseSeries, horizons: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for moment, value in zip(series.timestamps, series.values, strict=True):
        digest.update(f"{_iso(moment)}={value};".encode())
    digest.update(f"{series.interval_minutes}|{horizons}".encode())
    return digest.hexdigest()


def _iso(moment: datetime) -> str:
    return moment.isoformat()


class ForecasterCreator(ABC):
    """Creates the forecaster of one family member.

    Pattern: P02 Factory Method
    """

    @abstractmethod
    def create_forecaster(self) -> Forecaster: ...

    def create(self) -> Forecaster:
        """Every forecaster the service hands out is cached."""
        return CachingForecaster(self.create_forecaster())


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastOptions:
    """Hyperparameters of the forecasters, from the settings."""

    slope_window_minutes: int
    residual_window_points: int
    damping_per_hour: float
    spread_floor: float
    plausible_min: float
    plausible_max: float


class DampedTrendForecasterCreator(ForecasterCreator):
    def __init__(self, options: ForecastOptions) -> None:
        self._options = options

    @override
    def create_forecaster(self) -> Forecaster:
        o = self._options
        return DampedTrendForecaster(
            slope_window_minutes=o.slope_window_minutes,
            residual_window_points=o.residual_window_points,
            damping_per_hour=o.damping_per_hour,
            spread_floor=o.spread_floor,
            plausible_min=o.plausible_min,
            plausible_max=o.plausible_max,
        )


class LinearForecasterCreator(ForecasterCreator):
    def __init__(self, options: ForecastOptions) -> None:
        self._options = options

    @override
    def create_forecaster(self) -> Forecaster:
        o = self._options
        return LinearForecaster(
            slope_window_minutes=o.slope_window_minutes,
            spread_floor=o.spread_floor,
            plausible_min=o.plausible_min,
            plausible_max=o.plausible_max,
        )
