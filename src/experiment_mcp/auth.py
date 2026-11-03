"""JWT bearer verification for the MCP resource server.

The process validates tokens. It does not sign users in. ``mint_token`` exists
so local development can produce a token the same verifier accepts.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import anyio
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from jwt.algorithms import RSAAlgorithm
from mcp.server.auth.provider import AccessToken, TokenVerifier

from experiment_mcp.scopes import SUPPORTED_SCOPES

logger = logging.getLogger(__name__)

DEV_KEY_ID = "experiment-mcp-dev"


@dataclass(frozen=True)
class AuthConfig:
    issuer: str
    resource: str
    jwks_url: str | None = None
    jwks: dict[str, Any] | None = None


def generate_rsa_key() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def public_jwk(public_key: RSAPublicKey, kid: str = DEV_KEY_ID) -> dict[str, Any]:
    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [jwk]}


def normalize_scopes(requested: list[str]) -> list[str]:
    unknown = sorted(set(requested) - set(SUPPORTED_SCOPES))
    if unknown:
        raise ValueError(f"unknown scope(s): {', '.join(unknown)}")
    if not requested:
        raise ValueError("at least one scope is required")
    scopes = list(dict.fromkeys(requested))
    if "experiments:write" in scopes and "experiments:read" not in scopes:
        scopes.append("experiments:read")
    return scopes


def mint_token(
    *,
    private_key: RSAPrivateKey,
    issuer: str,
    resource: str,
    subject: str,
    scopes: list[str],
    ttl_seconds: int = 3600,
    kid: str = DEV_KEY_ID,
) -> str:
    if ttl_seconds < 60 or ttl_seconds > 24 * 3600:
        raise ValueError("ttl_seconds must be between 60 and 86400")
    issued_scopes = normalize_scopes(scopes)
    now = int(time.time())
    payload = {
        "iss": issuer,
        "sub": subject,
        "aud": resource,
        "iat": now,
        "exp": now + ttl_seconds,
        "scope": " ".join(issued_scopes),
        "client_id": subject,
    }
    token = jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})
    if isinstance(token, bytes):
        return token.decode()
    return token


def _loopback(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "::1"}


def validate_public_url(url: str, label: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an http(s) URL")
    if parsed.scheme != "https" and not _loopback(url):
        raise ValueError(f"{label} must use https unless the host is loopback")
    if parsed.fragment or parsed.query:
        raise ValueError(f"{label} must not include a query or fragment")


class JwtTokenVerifier(TokenVerifier):
    """Validate an RS256 access token against a JWKS URL or a static document.

    Failures return ``None`` so the HTTP layer responds with 401. The token
    itself is never logged.
    """

    def __init__(self, config: AuthConfig) -> None:
        if config.jwks is None and not config.jwks_url:
            raise ValueError("JWT verification needs a JWKS document or a JWKS URL")
        validate_public_url(config.issuer, "issuer")
        validate_public_url(config.resource, "resource")
        self._issuer = config.issuer
        self._resource = config.resource
        self._jwk_client = jwt.PyJWKClient(config.jwks_url, cache_keys=True) if config.jwks_url else None
        self._static = config.jwks

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            return await anyio.to_thread.run_sync(self._verify_sync, token)
        except Exception:
            logger.info("Rejected bearer token")
            return None

    def _verify_sync(self, token: str) -> AccessToken:
        key = self._signing_key(token)
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=self._resource,
            issuer=self._issuer,
            leeway=30,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
        raw_scope = payload.get("scope", "")
        if not isinstance(raw_scope, str):
            raise jwt.InvalidTokenError("scope claim must be a string")
        scopes = [scope for scope in raw_scope.split(" ") if scope]
        subject = str(payload["sub"])
        client_id = str(payload.get("client_id") or subject)
        expires_at = int(payload["exp"])
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=expires_at,
            resource=self._resource,
            subject=subject,
            claims={"iss": str(payload["iss"])},
        )

    def _signing_key(self, token: str) -> Any:
        if self._jwk_client is not None:
            return self._jwk_client.get_signing_key_from_jwt(token).key
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        keys = (self._static or {}).get("keys", [])
        for entry in keys:
            if kid is None or entry.get("kid") == kid:
                return jwt.PyJWK.from_dict(entry).key
        raise jwt.InvalidTokenError("no matching JWK")
