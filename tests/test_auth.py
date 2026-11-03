import json
import socket
import time
from collections.abc import AsyncIterator

import anyio
import jwt
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from starlette.testclient import TestClient

from experiment_mcp.auth import (
    AuthConfig,
    JwtTokenVerifier,
    generate_rsa_key,
    mint_token,
    public_jwk,
)
from experiment_mcp.config import serve_config_from_args
from experiment_mcp.dev_auth import build_dev_auth_app
from experiment_mcp.server import build_server, http_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def key_material() -> tuple[object, dict[str, object], str, str]:
    private = generate_rsa_key()
    issuer = "http://127.0.0.1:9000"
    resource = "http://127.0.0.1:8765/mcp"
    return private, public_jwk(private.public_key()), issuer, resource


def _token(private: object, issuer: str, resource: str, scopes: list[str], ttl: int = 600) -> str:
    return mint_token(
        private_key=private,  # type: ignore[arg-type]
        issuer=issuer,
        resource=resource,
        subject="tester",
        scopes=scopes,
        ttl_seconds=ttl,
    )


def test_verifier_accepts_and_rejects(
    key_material: tuple[object, dict[str, object], str, str],
) -> None:
    private, jwks, issuer, resource = key_material
    verifier = JwtTokenVerifier(AuthConfig(issuer=issuer, resource=resource, jwks=jwks))
    good = _token(private, issuer, resource, ["experiments:read"])

    async def check() -> None:
        accepted = await verifier.verify_token(good)
        assert accepted is not None
        assert accepted.scopes == ["experiments:read"]
        assert await verifier.verify_token(good + "x") is None
        wrong_aud = _token(private, issuer, "http://127.0.0.1:9/mcp", ["experiments:read"])
        assert await verifier.verify_token(wrong_aud) is None
        expired = jwt.encode(
            {
                "iss": issuer,
                "sub": "tester",
                "aud": resource,
                "iat": int(time.time()) - 120,
                "exp": int(time.time()) - 60,
                "scope": "experiments:read",
                "client_id": "tester",
            },
            private,  # type: ignore[arg-type]
            algorithm="RS256",
            headers={"kid": "experiment-mcp-dev"},
        )
        assert await verifier.verify_token(expired) is None

    anyio.run(check)


def test_http_metadata_and_rejection(
    key_material: tuple[object, dict[str, object], str, str], tmp_path: object
) -> None:
    private, jwks, issuer, resource = key_material
    from pathlib import Path

    server = build_server(
        db_path=Path(str(tmp_path)) / "experiments.db",
        auth=AuthConfig(issuer=issuer, resource=resource, jwks=jwks),
    )
    app = http_app(
        server,
        host="127.0.0.1",
        allowed_hosts=["127.0.0.1", "127.0.0.1:*", "localhost:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
    )
    with TestClient(app, base_url="http://127.0.0.1") as http:
        missing = http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert missing.status_code == 401
        assert "resource_metadata" in missing.headers["www-authenticate"]
        bad = http.post(
            "/mcp",
            headers={"authorization": "Bearer not-a-jwt"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        assert bad.status_code == 401
        metadata = http.get("/.well-known/oauth-protected-resource/mcp")
        assert metadata.status_code == 200
        body = metadata.json()
        assert body["resource"] == resource
        assert "experiments:read" in body["scopes_supported"]
        assert "experiments:write" in body["scopes_supported"]
        assert body["authorization_servers"]
        token = _token(private, issuer, resource, ["experiments:read"])
        allowed = http.post(
            "/mcp",
            headers={"authorization": f"Bearer {token}", "content-type": "application/json"},
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        assert allowed.status_code != 401


@pytest.fixture
async def authed_server(
    key_material: tuple[object, dict[str, object], str, str],
    tmp_path: object,
) -> AsyncIterator[tuple[str, object, str, str]]:
    from pathlib import Path

    private, jwks, issuer, _resource = key_material
    port = _free_port()
    resource = f"http://127.0.0.1:{port}/mcp"
    server = build_server(
        db_path=Path(str(tmp_path)) / "experiments.db",
        auth=AuthConfig(issuer=issuer, resource=resource, jwks=jwks),
    )
    app = http_app(
        server,
        host="127.0.0.1",
        allowed_hosts=["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    running = uvicorn.Server(config)
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(running.serve)
        for _ in range(100):
            if running.started:
                break
            await anyio.sleep(0.05)
        else:
            raise RuntimeError("uvicorn did not start")
        yield resource, private, issuer, resource
        running.should_exit = True


@pytest.mark.anyio
async def test_http_scope_acceptance(
    authed_server: tuple[str, object, str, str],
) -> None:
    import httpx2

    _url, private, issuer, resource = authed_server
    read_token = _token(private, issuer, resource, ["experiments:read"])
    write_token = _token(private, issuer, resource, ["experiments:write"])

    async def connect(token: str) -> Client:
        http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})
        transport = streamable_http_client(resource, http_client=http)
        return Client(transport, raise_exceptions=True)

    read_client = await connect(read_token)
    async with read_client:
        tools = await read_client.list_tools()
        assert any(tool.name == "power_calc" for tool in tools.tools)
        denied = await read_client.call_tool(
            "create_experiment",
            {"name": "Nope", "hypothesis": "Should fail.", "primary_metric": "redeemed"},
        )
        assert denied.is_error is True
        assert "experiments:write" in denied.content[0].text

    write_client = await connect(write_token)
    async with write_client:
        created = await write_client.call_tool(
            "create_experiment",
            {"name": "Allowed", "hypothesis": "Write scope.", "primary_metric": "redeemed"},
        )
        assert created.is_error is False
        listed = await write_client.call_tool("list_experiments", {})
        assert listed.structured_content["experiments"]


def test_dev_auth_requires_secret_and_mints(
    key_material: tuple[object, dict[str, object], str, str],
) -> None:
    _private, _jwks, _issuer, resource = key_material
    private = generate_rsa_key()
    app = build_dev_auth_app(issuer="http://127.0.0.1:9000", secret="dev-secret", private_key=private)
    with TestClient(app) as http:
        assert http.get("/.well-known/jwks.json").status_code == 200
        denied = http.post("/token", json={"resource": resource, "scope": "experiments:read"})
        assert denied.status_code == 401
        minted = http.post(
            "/token",
            headers={"authorization": "Bearer dev-secret"},
            json={"resource": resource, "scope": "experiments:write", "sub": "ada"},
        )
        assert minted.status_code == 200
        body = minted.json()
        assert "experiments:read" in body["scope"]
        assert "experiments:write" in body["scope"]
        verifier = JwtTokenVerifier(
            AuthConfig(
                issuer="http://127.0.0.1:9000",
                resource=resource,
                jwks=json.loads(http.get("/.well-known/jwks.json").text),
            )
        )

        async def check() -> None:
            token = await verifier.verify_token(body["access_token"])
            assert token is not None
            assert "experiments:write" in token.scopes

        anyio.run(check)


def test_http_without_auth_config_is_rejected(tmp_path: object) -> None:
    from argparse import Namespace

    with pytest.raises(SystemExit, match="HTTP transport requires"):
        serve_config_from_args(
            Namespace(transport="http", host="127.0.0.1", port=8000, db=None, data_dir=None),
            environ={},
        )
    config = serve_config_from_args(
        Namespace(transport="http", host="127.0.0.1", port=8000, db=str(tmp_path), data_dir=None),
        environ={"EXPERIMENT_MCP_ALLOW_INSECURE_NO_AUTH": "1"},
    )
    assert config.auth is None
    assert config.insecure_no_auth is True


@pytest.mark.anyio
async def test_stdio_transport_lists_tools(tmp_path: object) -> None:
    import os
    import sys

    from mcp.client.stdio import StdioServerParameters

    env = os.environ.copy()
    env["EXPERIMENT_MCP_DB"] = str(tmp_path) + "/stdio.db"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "experiment_mcp", "serve"],
        env=env,
    )
    async with Client(params) as client:
        tools = await client.list_tools()
        assert any(tool.name == "check_srm" for tool in tools.tools)
