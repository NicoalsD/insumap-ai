"""Service token check (Contrato-IA 2): constant time, current and previous token."""

from __future__ import annotations

import hmac
from typing import Final

from pydantic import SecretStr

BEARER: Final = "bearer "


class ServiceTokenVerifier:
    """Accepts ``AI_SERVICE_TOKEN`` and, during a rotation, the previous one."""

    def __init__(self, current: SecretStr, previous: SecretStr | None) -> None:
        self._tokens = [current.get_secret_value().encode()]
        if previous is not None:
            self._tokens.append(previous.get_secret_value().encode())

    def accepts(self, authorization: str | None) -> bool:
        if authorization is None or not authorization.lower().startswith(BEARER):
            return False
        presented = authorization[len(BEARER) :].strip().encode()
        # Compare with every token so the time does not reveal which one matched.
        matches = [hmac.compare_digest(presented, token) for token in self._tokens]
        return any(matches)
