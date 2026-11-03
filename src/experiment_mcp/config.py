"""Runtime configuration for the CLI.

Flags win over environment variables. HTTP refuses to start without a
token verifier unless insecure mode is set explicitly.
"""

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from experiment_mcp.auth import AuthConfig, validate_public_url

DEFAULT_HOSTS = ["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*", "[::1]", "[::1]:*"]
DEFAULT_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
    "https://127.0.0.1:*",
    "https://localhost:*",
]


@dataclass(frozen=True)
class ServeConfig:
    transport: Literal["stdio", "http"]
    host: str
    port: int
    db_path: Path
    data_dir: Path | None
    auth: AuthConfig | None
    allowed_hosts: list[str]
    allowed_origins: list[str]
    insecure_no_auth: bool


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _loopback_bind(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def resolve_auth(
    *,
    transport: Literal["stdio", "http"],
    insecure_no_auth: bool,
    environ: Mapping[str, str],
) -> AuthConfig | None:
    issuer = environ.get("EXPERIMENT_MCP_ISSUER_URL", "").strip()
    resource = environ.get("EXPERIMENT_MCP_RESOURCE_URL", "").strip()
    jwks_url = environ.get("EXPERIMENT_MCP_JWKS_URL", "").strip()
    jwks_path = environ.get("EXPERIMENT_MCP_JWKS_PATH", "").strip()
    if transport == "stdio":
        return None
    if insecure_no_auth:
        if issuer or resource or jwks_url or jwks_path:
            raise SystemExit(
                "Refusing to start. EXPERIMENT_MCP_ALLOW_INSECURE_NO_AUTH=1 cannot be combined "
                "with issuer, resource, or JWKS settings."
            )
        return None
    missing: list[str] = []
    if not issuer:
        missing.append("EXPERIMENT_MCP_ISSUER_URL")
    if not resource:
        missing.append("EXPERIMENT_MCP_RESOURCE_URL")
    if not jwks_url and not jwks_path:
        missing.append("EXPERIMENT_MCP_JWKS_URL or EXPERIMENT_MCP_JWKS_PATH")
    if missing:
        raise SystemExit(
            "HTTP transport requires OAuth bearer auth. Missing "
            + ", ".join(missing)
            + ". Set those, or EXPERIMENT_MCP_ALLOW_INSECURE_NO_AUTH=1 for a local throwaway process."
        )
    try:
        validate_public_url(issuer, "EXPERIMENT_MCP_ISSUER_URL")
        validate_public_url(resource, "EXPERIMENT_MCP_RESOURCE_URL")
        if jwks_url:
            validate_public_url(jwks_url, "EXPERIMENT_MCP_JWKS_URL")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    jwks = None
    if jwks_path:
        path = Path(jwks_path)
        if not path.is_file():
            raise SystemExit(f"JWKS file not found: {path}")
        jwks = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(jwks, dict) or "keys" not in jwks:
            raise SystemExit("JWKS file must be a JSON object with a keys array")
    return AuthConfig(issuer=issuer, resource=resource, jwks_url=jwks_url or None, jwks=jwks)


def _bind_host_allowed(host: str) -> None:
    if host in {"0.0.0.0", "::"}:
        return
    if not _loopback_bind(host):
        parsed = urlparse(f"http://{host}")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise SystemExit(
                f"Refusing to bind HTTP to {host!r}. Use 127.0.0.1, or 0.0.0.0 inside a container "
                "and set EXPERIMENT_MCP_ALLOWED_HOSTS to the Host headers you expect."
            )


def serve_config_from_args(args: argparse.Namespace, environ: dict[str, str] | None = None) -> ServeConfig:
    env = os.environ if environ is None else environ
    transport: Literal["stdio", "http"] = args.transport
    insecure = env.get("EXPERIMENT_MCP_ALLOW_INSECURE_NO_AUTH") == "1"
    db_value = args.db or env.get("EXPERIMENT_MCP_DB") or str(Path.cwd() / ".experiment_mcp" / "experiments.db")
    data_value = args.data_dir or env.get("EXPERIMENT_MCP_DATA_DIR")
    host = args.host or env.get("EXPERIMENT_MCP_HOST") or "127.0.0.1"
    port = int(args.port or env.get("EXPERIMENT_MCP_PORT") or "8000")
    if transport == "http":
        _bind_host_allowed(host)
    auth = resolve_auth(transport=transport, insecure_no_auth=insecure, environ=env)
    extra_hosts = _csv(env.get("EXPERIMENT_MCP_ALLOWED_HOSTS"))
    extra_origins = _csv(env.get("EXPERIMENT_MCP_ALLOWED_ORIGINS"))
    return ServeConfig(
        transport=transport,
        host=host,
        port=port,
        db_path=Path(db_value),
        data_dir=Path(data_value) if data_value else None,
        auth=auth,
        allowed_hosts=[*DEFAULT_HOSTS, *extra_hosts],
        allowed_origins=[*DEFAULT_ORIGINS, *extra_origins],
        insecure_no_auth=insecure and transport == "http",
    )
