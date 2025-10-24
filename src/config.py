from pathlib import Path

# --- パス ---
_CONFIG_FILEPATH = Path(__file__).resolve()
SRC_ROOT = _CONFIG_FILEPATH.parent
PROJECT_ROOT = SRC_ROOT.parent

# --- ディレクトリ ---
DATA_DIR = PROJECT_ROOT / "data"
DEMONSTRATION_DIR = DATA_DIR / "demonstrations"
