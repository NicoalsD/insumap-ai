"""The single error format of the service (Contrato-IA 7).

``code`` in English, ``message`` in Spanish, never data from the request.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


class ServiceError(Exception):
    """An error the contract names, rendered as its JSON envelope."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers


class InvalidSeriesError(ServiceError):
    """``422 INVALID_SERIES`` with the reason the contract lists."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            422,
            "INVALID_SERIES",
            "La serie de glucosa no es válida.",
            {"reason": reason},
        )


class LimitExceededError(ServiceError):
    """``422 VALIDATION_ERROR`` for a technical limit of the settings."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            422,
            "VALIDATION_ERROR",
            "La petición no cumple el contrato.",
            {"fields": [{"path": path, "reason": reason}]},
        )


def error_response(error: ServiceError, request_id: str) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {"code": error.code, "message": error.message},
        "request_id": request_id,
    }
    if error.details is not None:
        body["error"]["details"] = error.details
    return JSONResponse(body, status_code=error.status, headers=error.headers)
