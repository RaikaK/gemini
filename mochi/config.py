# config.py - 最小限の設定ファイル

import os

# --- APIモデル設定 ---
# 環境変数から取得、なければプレースホルダー
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

# 使用するモデル名
INTERVIEWER_MODEL = "gpt-4o-mini"  # 面接官役
APPLICANT_MODEL = "gpt-4o-mini"     # 応募者役

# --- 実験設定 ---
NUM_CANDIDATES = 3  # 候補者の数
MAX_ROUNDS = 5      # 面接ラウンド数

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
ENABLE_SPREADSHEET = False  # スプレッドシート連携を有効にするかどうか
