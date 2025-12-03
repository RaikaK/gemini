# config.py - 最小限の設定ファイル

import os

# --- APIモデル設定 ---
# 環境変数から取得、なければプレースホルダー
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

# APIキーの検証（起動時に警告を表示）
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE" or not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
    print("=" * 80)
    print("警告: OPENAI_API_KEYが設定されていません")
    print("=" * 80)
    print("環境変数を設定してください:")
    print("  Windows PowerShell:")
    print("    $env:OPENAI_API_KEY='your-api-key-here'")
    print("  Windows CMD:")
    print("    set OPENAI_API_KEY=your-api-key-here")
    print("  Linux/Mac:")
    print("    export OPENAI_API_KEY='your-api-key-here'")
    print("=" * 80)
    print()

# 使用するモデル名
INTERVIEWER_MODEL = "gpt-4o-mini"  # 面接官役
APPLICANT_MODEL = "gpt-4o-mini"     # 応募者役

# --- 実験設定 ---
NUM_CANDIDATES = 3  # 候補者の数
MAX_ROUNDS = 20     # 面接ラウンド数（デフォルト）

# 知識レベルの設定
KNOWLEDGE_RETENTION_RATIO = {
    'low': 0.2,     # 志望度が低い候補者の知識保持率
    'medium': 0.5,  # 志望度が中程度の候補者の知識保持率
    'high': 1.0     # 志望度が高い候補者の知識保持率（全て）
}

# 回答の文字数制限
MAX_ANSWER_LENGTH = 150  # 学生の回答の最大文字数

# 志望度レベルのマッピング
ASPIRATION_LEVEL_MAPPING = {
    'high_90_percent': 'high',
    'medium_70_percent': 'medium',
    'low_50_percent': 'low',
}

# データベースファイルのパス
DB_FILE_PATH = "../experiment_inter/db.json"  # mochiディレクトリから見た相対パス

# デフォルト値
DEFAULT_KNOWLEDGE_RETENTION_RATIO = 0.2  # 不明な志望度レベルのデフォルト保持率

# --- 複数回シミュレーション設定 ---
NUM_SIMULATIONS = 1  # デフォルトのシミュレーション実行回数

# --- スプレッドシート連携設定 ---
ENABLE_SPREADSHEET = True  # スプレッドシート連携を有効にするかどうか

# --- ローカルモデル設定 ---
# 面接官モデルタイプ: 'api' または 'local'
INTERVIEWER_MODEL_TYPE = 'api'  # デフォルトはAPI

# 利用可能なローカルモデル（model_manager.pyのavailable_modelsと一致）
# モデルタイプの分類:
# - "llama3": Llama-3系モデル（Llama-3.1, Llama-3など）
# - "llama2": Llama-2系モデル（ELYZA-japanese-Llama-2など）
# - "other": その他のモデル（Qwen, TinyLlamaなど）
AVAILABLE_LOCAL_MODELS = {
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "SWALLOW": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
    "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    # "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # 日本語がまともに出力されないのでなし。　
    # "ELYZA-japanese-Llama-2": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    # "llama3-elyza-jp": "elyza/Llama-3-ELYZA-JP-8B",
}

# モデルタイプのマッピング（チャットテンプレートの形式を決定）
MODEL_TYPE_MAPPING = {
    "llama3": "llama3",
    "SWALLOW": "llama3",
    "qwen3-4b-instruct-2507": "qwen",
    "ELYZA-japanese-Llama-2": "llama2",
    "llama3-elyza-jp": "llama3",
    # "tinyllama": "llama2",  # TinyLlamaはLlama-2ベースのためllama2形式を使用
    "qwen2.5-7b-instruct": "qwen",
}

# デフォルトのローカルモデル
LOCAL_MODEL_NAME = "llama3"  # AVAILABLE_LOCAL_MODELSのキーを指定

# --- wandb設定 ---
ENABLE_WANDB = True  # wandbログを有効にするかどうか
WANDB_PROJECT = "penguin-paper-interviews"  # wandbプロジェクト名
WANDB_ENTITY = None  # wandbエンティティ（Noneの場合はデフォルト）
