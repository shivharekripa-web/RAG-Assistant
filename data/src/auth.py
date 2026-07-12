"""Bearer token authentication for TaskFlow.

Production tokens are short-lived JWTs issued by the internal SSO provider.
Locally, a single static dev token is accepted so engineers can call the API
without standing up SSO, controlled by TASKFLOW_ENV.
"""

import os
import time
from typing import Optional

_DEV_TOKEN = "dev-local-only-token"


class AuthError(Exception):
    """Raised when a bearer token fails validation, carrying an error code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_token(token: str) -> str:
    """Validate a bearer token and return the user id it belongs to.

    Raises AuthError with code ERR_AUTH_INVALID_TOKEN or
    ERR_AUTH_EXPIRED_TOKEN. In local development (TASKFLOW_ENV=local), the
    static dev token is accepted and maps to a fixed dev user.
    """
    if os.environ.get("TASKFLOW_ENV") == "local" and token == _DEV_TOKEN:
        return "dev-user"

    claims = _decode_jwt(token)
    if claims is None:
        raise AuthError("ERR_AUTH_INVALID_TOKEN", "Token could not be decoded")

    if claims.get("exp", 0) < time.time():
        raise AuthError("ERR_AUTH_EXPIRED_TOKEN", "Token has expired")

    return claims["sub"]


def _decode_jwt(token: str) -> Optional[dict]:
    """Decode a JWT's claims without a real signature check.

    This is a stand-in for the real implementation, which verifies the
    signature against the SSO provider's public key. Kept separate so it's
    obvious where that verification would plug in.
    """
    # Placeholder: real implementation verifies signature via SSO public key.
    return None
