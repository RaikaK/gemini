# config.py

# --- APIモデル設定 (データ生成 & 学生役 & API面接官役) ---
OPENAI_API_KEY = "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"  # ご自身のAPIキーを設定してください
GENERATOR_MODEL_NAME = "gpt-4o-mini"
APPLICANT_API_MODEL = "gpt-4o-mini" # 学生役として使用するAPIモデル

# --- ローカルモデル設定 (ローカル面接官役) ---
# 利用可能なローカルモデル一覧
AVAILABLE_LOCAL_MODELS = {
    # 主要な日本語対応モデル
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "ELYZA-japanese-Llama-2": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    "SWALLOW": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
    "llama3-elyza-jp": "elyza/Llama-3-ELYZA-JP-8B",
    
    # Llama 3シリーズ
    "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    
    # 日本語特化モデル
    "japanese-stablelm": "stabilityai/japanese-stablelm-instruct-gamma-7b",
    "weblab-10b": "rinna/weblab-10b-instruction-sft",
    "calm2-7b": "cyberagent/calm2-7b-chat",
    "calm2-3b": "cyberagent/calm2-3b-chat",
    
    # 軽量・高性能モデル
    "gemma2-9b": "google/gemma-2-9b-it",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # その他の高性能モデル
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen-7B-Chat",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct"
}

# デフォルトのローカルモデル
LOCAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# --- 面接官モデル選択 ---
# 'local' を指定するとローカルモデル (Llama) が面接官になります。
# 'api' を指定するとOpenAI APIモデル (GPT) が面接官になります。
INTERVIEWER_MODEL_TYPE = 'api'  # 'local' or 'api'
# 利用可能なAPIモデル一覧
AVAILABLE_API_MODELS = {
    "chatgpt4o": "gpt-4o",
    "chatgpt4": "gpt-4",
    "chatgpt4o-mini": "gpt-4o-mini",
    "chatgpt4-turbo": "gpt-4-turbo",
    "chatgpt3.5-turbo": "gpt-3.5-turbo",
    "chatgpt5": "gpt-5"  # 将来のリリース用
}

INTERVIEWER_API_MODEL = "gpt-4o-mini" # 面接官を 'api' に設定した場合に使用するモデル

# --- 実験設定 ---
NUM_CANDIDATES = 3 # 生成する学生の数

# --- 面接フローの設定 ---
# 面接の流れを定義（0: 全体質問, 1: 個別質問）
# 例: [0, 1, 1, 1] = 全体質問1回 + 個別質問3回
INTERVIEW_FLOW = [1,1,1,1,1]  # 全体質問1回 + 個別質問3回

# --- 智的動的面接フローの設定 ---
# 智的動的フローを使用するかどうか
USE_INTELLIGENT_DYNAMIC_FLOW = True  # True: 智的動的フロー, False: 固定フロー
MAX_DYNAMIC_ROUNDS = 10  # 動的フローでの最大ラウンド数（質問回数制限撤廃に対応）

# --- 対話設定 ---
MAX_CONVERSATION_TURNS = 10  # 1回の面接での最大対話回数
MIN_CONVERSATION_TURNS = 1   # 1回の面接での最小対話回数