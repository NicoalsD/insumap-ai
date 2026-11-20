"""FastAPI application of the AI service (Contrato-IA 1, 2, 7 and 9)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated, Any, Final

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from insumap_ai.config import Settings
from insumap_ai.contract import (
    CONTRACT_VERSION,
    AssistantReplyRequest,
    AssistantReplyResponse,
    ClinicalSummaryRequest,
    ClinicalSummaryResponse,
    GlucoseForecastRequest,
    GlucoseForecastResponse,
    ModelsResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
)
from insumap_ai.errors import ServiceError, error_response
from insumap_ai.pipeline import (
    AssistantPipeline,
    ForecastPipeline,
    RiskPipeline,
    SummaryPipeline,
)
from insumap_ai.registry import ModelRegistry
from insumap_ai.security import ServiceTokenVerifier

REQUEST_ID_HEADER: Final = "X-Request-Id"
RETRY_AFTER_SECONDS: Final = 5
logger = logging.getLogger("insumap_ai")


class BodyTooLargeError(Exception):
    pass


class BodyLimitMiddleware:
    """Refuses bodies above ``MAX_BODY_BYTES`` without buffering them first."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max:
            await self._too_large(scope, receive, send)
            return
        seen = 0

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self._max:
                    raise BodyTooLargeError
            return message

        try:
            await self._app(scope, counted, send)
        except BodyTooLargeError:
            await self._too_large(scope, receive, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        error = ServiceError(
            413,
            "PAYLOAD_TOO_LARGE",
            "El cuerpo de la petición supera el tamaño permitido.",
            {"max_bytes": self._max},
        )
        response = error_response(error, _request_id_of(scope))
        await response(scope, receive, send)


def _request_id_of(scope: Scope) -> str:
    raw = dict(scope.get("headers", [])).get(REQUEST_ID_HEADER.lower().encode())
    if raw:
        try:
            return str(uuid.UUID(raw.decode()))
        except ValueError:
            pass
    return str(uuid.uuid4())


def request_id(request: Request) -> str:
    value: str = request.state.request_id
    return value


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    verifier = ServiceTokenVerifier(
        config.ai_service_token, config.ai_service_token_previous
    )
    registry = ModelRegistry.instance()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            registry.load(config)
        except Exception as error:  # /v1/ready reports it
            logger.error("models failed to load: %s", type(error).__name__)
        yield

    docs = config.api_docs_enabled
    app = FastAPI(
        title="Insumap IA",
        description=(
            "Servicio de IA sin estado de Insumap. Proyecto académico: no es un "
            "dispositivo médico y la IA solo apoya al médico."
        ),
        version=CONTRACT_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = _request_id_of(request.scope)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

    app.add_middleware(BodyLimitMiddleware, max_bytes=config.max_body_bytes)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError) -> JSONResponse:
        return error_response(error, request_id(request))

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        fields = [
            {
                "path": ".".join(str(part) for part in item["loc"][1:]),
                "reason": str(item["type"]).upper(),
            }
            for item in error.errors()
        ]
        return error_response(
            ServiceError(
                422,
                "VALIDATION_ERROR",
                "La petición no cumple el contrato.",
                {"fields": fields},
            ),
            request_id(request),
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        logger.error(
            "unexpected error",
            extra={"request_id": request_id(request), "error": type(error).__name__},
        )
        return error_response(
            ServiceError(500, "INTERNAL_ERROR", "Error interno del servicio de IA."),
            request_id(request),
        )

    def authorized(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not verifier.accepts(authorization):
            raise ServiceError(
                401,
                "INVALID_SERVICE_TOKEN",
                "Token de servicio ausente o inválido.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def ready() -> None:
        if not registry.ready:
            raise ServiceError(
                503,
                "MODEL_NOT_READY",
                "Los modelos aún no están listos.",
                {"retry_after_seconds": RETRY_AFTER_SECONDS},
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )

    guarded: Any = [Depends(authorized), Depends(ready)]
    tag: list[str | Enum] = ["Inferencia"]

    @app.get("/v1/health", tags=["Salud"], summary="Proceso vivo")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready", tags=["Salud"], summary="Modelos cargados")
    def readiness() -> JSONResponse:
        if registry.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse(
            {"status": "not_ready"},
            status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )

    @app.get(
        "/v1/models",
        tags=["Salud"],
        summary="Modelos cargados, versiones y estado",
        dependencies=guarded,
        response_model_by_alias=True,
    )
    def models() -> ModelsResponse:
        return ModelsResponse(
            contract_version=CONTRACT_VERSION,
            provider_family=registry.models.family,
            models=registry.cards(config),
        )

    @app.post(
        "/v1/glucose-forecasts",
        tags=tag,
        summary="IA1: pronóstico de glucosa y riesgo de hipo o hiperglucemia",
        dependencies=guarded,
        response_model_by_alias=True,
    )
    def glucose_forecasts(
        body: GlucoseForecastRequest, request: Request
    ) -> GlucoseForecastResponse:
        m = registry.models
        pipeline = ForecastPipeline(config, m.forecaster, m.fallback_forecaster)
        return pipeline.run(body, request_id(request))

    @app.post(
        "/v1/risk-assessments",
        tags=tag,
        summary="IA2 e IA7: riesgo de deterioro o de abandono",
        dependencies=guarded,
        response_model_by_alias=True,
    )
    def risk_assessments(
        body: RiskAssessmentRequest, request: Request
    ) -> RiskAssessmentResponse:
        return RiskPipeline(config, registry.models.risk).run(body, request_id(request))

    @app.post(
        "/v1/clinical-summaries",
        tags=tag,
        summary="IA5: resumen clínico previo a la consulta",
        dependencies=guarded,
        response_model_by_alias=True,
    )
    def clinical_summaries(
        body: ClinicalSummaryRequest, request: Request
    ) -> ClinicalSummaryResponse:
        pipeline = SummaryPipeline(config, registry.models.text)
        return pipeline.run(body, request_id(request))

    @app.post(
        "/v1/assistant-replies",
        tags=tag,
        summary="IA6: respuesta del asistente educativo con guardrails",
        dependencies=guarded,
        response_model_by_alias=True,
    )
    def assistant_replies(
        body: AssistantReplyRequest, request: Request
    ) -> AssistantReplyResponse:
        pipeline = AssistantPipeline(config, registry.models.text)
        return pipeline.run(body, request_id(request))

    return app


def app() -> FastAPI:
    """Factory for ``uvicorn --factory insumap_ai.main:app``."""
    return create_app()
