from pathlib import Path

# ==============================================================
# パス
# ==============================================================
_CONFIG_FILEPATH = Path(__file__).resolve()
SRC_ROOT = _CONFIG_FILEPATH.parent
PROJECT_ROOT = SRC_ROOT.parent

# ===============================================================
# ディレクトリ
# ==============================================================
DATA_DIR = PROJECT_ROOT / "data"
DEMONSTRATION_DIR = DATA_DIR / "demonstrations"
MODELS_DIR = PROJECT_ROOT / "models"

# ===============================================================
# 特徴量
# ===============================================================
CHANNELS_CARD = 55  # チャンネル数 (カード)
CHANNELS_GENERAL = 33  # チャンネル数 (局面)
CHANNELS_CHAIN = 12  # チャンネル数 (チェーン)
CHANNELS_SELECTION = 22  # チャンネル数 (行動要求)
CHANNELS_ENTRY = 53  # チャンネル数 (行動選択)
CHANNELS_LOG = CHANNELS_SELECTION + CHANNELS_ENTRY  # チャンネル数 (行動ログ)
MAX_LOG_HISTORY = 3  # 最大履歴数 (行動ログ)

TOTAL_CHANNELS_STATE = (
    CHANNELS_CARD
    + CHANNELS_GENERAL
    + CHANNELS_CHAIN
    + CHANNELS_SELECTION
    + (CHANNELS_LOG * MAX_LOG_HISTORY)
    + CHANNELS_ENTRY
)  # 合計チャンネル数 (状態)
TOTAL_CHANNELS_ACTION = CHANNELS_ENTRY  # 合計チャンネル数 (行動)
TOTAL_CHANNELS_STATE_ACTION = TOTAL_CHANNELS_STATE + TOTAL_CHANNELS_ACTION  # 合計チャンネル数 (状態＋行動)
HEIGHT = 8  # 高さ
WIDTH = 5  # 幅

# ===============================================================
# WandB
# ===============================================================
WANDB_ENTITY = "ygo-ai"
WANDB_PROJECT = "U-Ni-Yo"
