# config.py

# --- APIモデル設定 (データ生成 & 学生役 & API面接官役) ---
OPENAI_API_KEY = "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"  # ご自身のAPIキーを設定してください
GENERATOR_MODEL_NAME = "gpt-4o-mini"
APPLICANT_API_MODEL = "gpt-4o-mini" # 学生役として使用するAPIモデル

# --- ローカルモデル設定 (ローカル面接官役) ---
LOCAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOCAL_MODEL_NAME = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"

# --- 面接官モデル選択 ---
# 'local' を指定するとローカルモデル (Llama) が面接官になります。
# 'api' を指定するとOpenAI APIモデル (GPT) が面接官になります。
INTERVIEWER_MODEL_TYPE = 'local'  # 'local' or 'api'
INTERVIEWER_API_MODEL = "gpt-4o-mini" # 面接官を 'api' に設定した場合に使用するモデル

# --- 実験設定 ---
NUM_CANDIDATES = 3 # 生成する学生の数

# --- 面接フローの設定 ---
# 面接の流れを定義（0: 全体質問, 1: 個別質問）
# 例: [0, 1, 1, 1] = 全体質問1回 + 個別質問3回
INTERVIEW_FLOW = [1,1,1,1,1]  # 全体質問1回 + 個別質問3回

# --- 対話設定 ---
MAX_CONVERSATION_TURNS = 10  # 1回の面接での最大対話回数
MIN_CONVERSATION_TURNS = 1   # 1回の面接での最小対話回数