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
