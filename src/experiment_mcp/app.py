"""Default server object for `mcp dev` and the MCP Inspector.

Importing this module uses the SQLite file and dataset selected by the
environment, same as `experiment-mcp serve`.
"""

import os
from pathlib import Path

from experiment_mcp.dataset import packaged_data_dir
from experiment_mcp.server import build_server

_db = Path(os.environ.get("EXPERIMENT_MCP_DB", Path.cwd() / ".experiment_mcp" / "experiments.db"))
_data = os.environ.get("EXPERIMENT_MCP_DATA_DIR")
mcp = build_server(db_path=_db, data_dir=Path(_data) if _data else packaged_data_dir())
