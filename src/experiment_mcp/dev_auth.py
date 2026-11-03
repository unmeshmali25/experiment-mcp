"""Localhost token minter for development.

This is not a production authorization server. It does not sign a person in,
does not implement the authorization-code grant, and refuses to bind to a
non-loopback address. The MCP server still validates the resulting JWTs itself.
"""

import argparse
import hmac
import logging
import os
import secrets
from collections.abc import Mapping
from typing import Any

import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from experiment_mcp.auth import (
    DEV_KEY_ID,
    generate_rsa_key,
    mint_token,
    public_jwk,
    validate_public_url,
)
from experiment_mcp.scopes import SUPPORTED_SCOPES

logger = logging.getLogger(__name__)


def _issuer(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _authorized(request: Request, secret: str) -> bool:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return False
    presented = header[7:].strip()
    return hmac.compare_digest(presented, secret)


def build_dev_auth_app(*, issuer: str, secret: str, private_key: RSAPrivateKey) -> Starlette:
    jwks = public_jwk(private_key.public_key(), DEV_KEY_ID)

    async def health(_: Request) -> PlainTextResponse:
        return PlainTextResponse(
            "experiment-mcp dev token minter. Not a production authorization server. "
            "POST /token with the dev secret to mint a JWT.\n"
        )

    async def jwks_route(_: Request) -> JSONResponse:
        return JSONResponse(jwks)

    async def metadata(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "issuer": issuer,
                "jwks_uri": f"{issuer}/.well-known/jwks.json",
                "token_endpoint": f"{issuer}/token",
                "scopes_supported": list(SUPPORTED_SCOPES),
                "experiment_mcp_dev_only": True,
            }
        )

    async def token(request: Request) -> JSONResponse:
        if not _authorized(request, secret):
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        try:
            body: Mapping[str, Any] = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "invalid_request", "error_description": "JSON body required"},
                status_code=400,
            )
        raw_scope = body.get("scope", "experiments:read")
        if isinstance(raw_scope, str):
            requested = [part for part in raw_scope.split() if part]
        elif isinstance(raw_scope, list) and all(isinstance(part, str) for part in raw_scope):
            requested = list(raw_scope)
        else:
            return JSONResponse({"error": "invalid_scope"}, status_code=400)
        resource = body.get("resource")
        subject = body.get("sub", "local-dev")
        ttl = body.get("ttl_seconds", 3600)
        if not isinstance(resource, str) or not isinstance(subject, str) or not isinstance(ttl, int):
            return JSONResponse(
                {
                    "error": "invalid_request",
                    "error_description": "resource (url), sub (str), ttl_seconds (int)",
                },
                status_code=400,
            )
        try:
            validate_public_url(resource, "resource")
            access_token = mint_token(
                private_key=private_key,
                issuer=issuer,
                resource=resource,
                subject=subject,
                scopes=requested,
                ttl_seconds=ttl,
            )
        except ValueError as exc:
            return JSONResponse({"error": "invalid_request", "error_description": str(exc)}, status_code=400)
        from experiment_mcp.auth import normalize_scopes

        scopes = normalize_scopes(requested)
        logger.info("Minted dev token for sub=%s scopes=%s", subject, " ".join(scopes))
        return JSONResponse(
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": ttl,
                "scope": " ".join(scopes),
            }
        )

    return Starlette(
        routes=[
            Route("/", health, methods=["GET"]),
            Route("/.well-known/jwks.json", jwks_route, methods=["GET"]),
            Route("/.well-known/oauth-authorization-server", metadata, methods=["GET"]),
            Route("/token", token, methods=["POST"]),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Localhost JWT minter for experiment-mcp.")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("dev-auth only binds to loopback. It is not a production identity provider.")
    secret = os.environ.get("EXPERIMENT_MCP_DEV_AUTH_SECRET") or secrets.token_urlsafe(24)
    private_key = generate_rsa_key()
    issuer = _issuer(args.port)
    app = build_dev_auth_app(issuer=issuer, secret=secret, private_key=private_key)
    print("experiment-mcp dev authorization server")
    print("This process mints tokens. It does not authenticate end users.")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"JWKS: {issuer}/.well-known/jwks.json")
    print("Send the dev secret as Authorization: Bearer <secret> to POST /token.")
    print(f"EXPERIMENT_MCP_DEV_AUTH_SECRET={secret}")
    print("Point the MCP server at:")
    print(f"  EXPERIMENT_MCP_ISSUER_URL={issuer}")
    print(f"  EXPERIMENT_MCP_JWKS_URL={issuer}/.well-known/jwks.json")
    # The private key stays in memory. Print the public key only.
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    logger.debug("Dev auth public key:\n%s", public_pem.decode())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
