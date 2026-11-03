"""OAuth scopes enforced on the HTTP transport.

``experiments:read`` covers tools that only compute or look up state.
``experiments:write`` covers tools that create or close experiment records.
A write token is also allowed to call read tools. stdio has no token, so
these checks are skipped there.
"""

READ_SCOPE = "experiments:read"
WRITE_SCOPE = "experiments:write"
SUPPORTED_SCOPES: tuple[str, ...] = (READ_SCOPE, WRITE_SCOPE)
