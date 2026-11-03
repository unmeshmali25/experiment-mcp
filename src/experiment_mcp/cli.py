"""Command line entry points."""

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn
from cryptography.hazmat.primitives import serialization

from experiment_mcp.auth import (
    DEV_KEY_ID,
    generate_rsa_key,
    mint_token,
    public_jwk,
    validate_public_url,
)
from experiment_mcp.config import serve_config_from_args
from experiment_mcp.server import build_server, http_app


def _serve(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="experiment-mcp serve")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    config = serve_config_from_args(args)
    if config.insecure_no_auth:
        logging.getLogger(__name__).warning(
            "HTTP auth is disabled. Do not expose this process beyond your own machine."
        )
    server = build_server(db_path=config.db_path, data_dir=config.data_dir, auth=config.auth)
    if config.transport == "stdio":
        server.run(transport="stdio")
        return
    app = http_app(
        server,
        host=config.host,
        allowed_hosts=config.allowed_hosts,
        allowed_origins=config.allowed_origins,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


def _mint(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="experiment-mcp mint-token")
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--sub", default="local-dev")
    parser.add_argument("--scope", action="append", required=True, help="Repeat for each scope.")
    parser.add_argument("--ttl-seconds", type=int, default=3600)
    parser.add_argument("--jwks-out", type=Path, required=True)
    parser.add_argument("--private-key-out", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_public_url(args.issuer, "issuer")
    validate_public_url(args.resource, "resource")
    private_key = generate_rsa_key()
    args.private_key_out.parent.mkdir(parents=True, exist_ok=True)
    args.jwks_out.parent.mkdir(parents=True, exist_ok=True)
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    args.private_key_out.write_bytes(pem)
    args.private_key_out.chmod(0o600)
    args.jwks_out.write_text(
        json.dumps(public_jwk(private_key.public_key(), DEV_KEY_ID), indent=2) + "\n",
        encoding="utf-8",
    )
    token = mint_token(
        private_key=private_key,
        issuer=args.issuer,
        resource=args.resource,
        subject=args.sub,
        scopes=args.scope,
        ttl_seconds=args.ttl_seconds,
    )
    print(token)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: experiment-mcp {serve,mint-token,dev-auth} ...", file=sys.stderr)
        print("  serve        stdio (default) or --transport http", file=sys.stderr)
        print("  mint-token   write a dev RSA key, JWKS, and print a JWT", file=sys.stderr)
        print("  dev-auth     localhost token minter (not a production IdP)", file=sys.stderr)
        raise SystemExit(0 if args and args[0] in {"-h", "--help"} else 2)
    command, rest = args[0], args[1:]
    if command == "serve":
        _serve(rest)
        return
    if command == "mint-token":
        _mint(rest)
        return
    if command == "dev-auth":
        from experiment_mcp.dev_auth import main as dev_auth_main

        dev_auth_main(rest)
        return
    raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
