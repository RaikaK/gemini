# --- APIモデル設定 (面接官役 & データ生成役) ---
GEMINI_API_KEY = "AIzaSyA_XleL8lGvzJAE1QTpfS429amLos6jqgc"
INTERVIEWER_MODEL_NAME = "gemini-1.5-flash"
GENERATOR_MODEL_NAME = "gemini-1.5-flash" 

# --- ローカルモデル設定 (学生役) ---
LOCAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# --- 実験設定 ---
NUM_CANDIDATES = 3 # 生成する学生の数

# --- 面接フローの設定 (0: 全体質問, 1: 個別質問) ---
# 例: [0, 1, 0] -> 全体質問 → 個別質問 → 全体質問
INTERVIEW_FLOW = [0, 1, 0] 

# 全体質問で使用する質問のリスト
COMMON_QUESTIONS = [
    "自己紹介と、当社を志望した理由を教えてください。",
    "学生時代に最も力を入れたことは何ですか？その経験から何を学びましたか？",
    "入社後、あなたは当社でどのように貢献できると考えますか？",
    "あなたの長所と短所を教えてください。",
    "最後に何か質問はありますか？"
]
