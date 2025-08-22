# config.py

# --- APIモデル設定 (データ生成 & 今回は【学生役】) ---
OPENAI_API_KEY = "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"  # ご自身のAPIキーを設定してください
GENERATOR_MODEL_NAME = "gpt-4o-mini"
APPLICANT_API_MODEL = "gpt-4o-mini" # 学生役として使用するAPIモデル

# --- ローカルモデル設定 (今回は【面接官役】) ---
# main.pyで初期化され、面接官として振る舞います
LOCAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# --- 実験設定 ---
NUM_CANDIDATES = 3 # 生成する学生の数

# --- 面接フローの設定 (0: 全体質問, 1: 個別質問) ---
# 例: [0, 1, 0] -> 全体質問 → 個別質問 → 全体質問
INTERVIEW_FLOW = [0, 1, 0] 

# COMMON_QUESTIONS = [
#     "自己紹介と、当社を志望した理由を教えてください。",
#     "学生時代に最も力を入れたことは何ですか？その経験から何を学びましたか？",
#     "入社後、あなたは当社でどのように貢献できると考えますか？",
#     "あなたの長所と短所を教えてください。",
#     "最後に何か質問はありますか？"
# ]