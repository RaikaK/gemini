# run_experiment.py (統合最終版)

import os
import json
import datetime
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import google.generativeai as genai
import random
import re

# ==============================================================================
# 1. 設定 (Config)
# ==============================================================================
# --- APIモデル設定 (面接官役 & データ生成役) ---
# TODO: ご自身のAPIキーを設定してください
GEMINI_API_KEY = "AIzaSyA_XleL8lGvzJAE1QTpfS429amLos6jqgc"
INTERVIEWER_MODEL_NAME = "gemini-1.5-flash"
GENERATOR_MODEL_NAME = "gemini-1.5-flash" 

# --- ローカルモデル設定 (学生役) ---
LOCAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# --- 実験設定 ---
NUM_CANDIDATES = 3 # 生成する学生の数

# ★ 新設: 面接フローの設定 (0: 全体質問, 1: 個別質問)
INTERVIEW_FLOW = [0, 1, 0] 
COMMON_QUESTIONS = [
    "自己紹介と、当社を志望した理由を教えてください。",
    "学生時代に最も力を入れたことは何ですか？その経験から何を学びましたか？",
    "入社後、あなたは当社でどのように貢献できると考えますか？",
    "あなたの長所と短所を教えてください。",
    "最後に何か質問はありますか？"
]


# ==============================================================================
# 2. ユーティリティ関数
# ==============================================================================
def call_gemini_api(model_name, prompt):
    """Gemini APIを呼び出し、テキスト応答を返す"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        error_message = "APIキーが設定されていません。処理を中断します。"
        print(f"エラー: {error_message}")
        if "JSON" in prompt:
            return json.dumps({"error": error_message})
        return error_message
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API呼び出しエラー: {e}"

def parse_json_from_response(response_text):
    """LLMの応答からJSON部分を抽出してパースする"""
    try:
        if '```json' in response_text:
            json_string = response_text.split('```json')[1].split('```')[0]
        else:
            json_string = response_text
        return json.loads(json_string.strip())
    except (json.JSONDecodeError, IndexError) as e:
        print(f"JSONのパースに失敗しました。エラー: {e}")
        return {"error": "Failed to parse JSON", "raw_output": response_text}

# ==============================================================================
# 3. 動的情報生成 (Gemini API)
# ==============================================================================
def generate_company_profile():
    """架空の企業情報を生成する"""
    print("--- 企業情報の生成を開始 ---")
    prompt = """
    日本の架空の企業1社の情報を生成してください。

    以下のJSON形式で出力してください：

    ```json
    {
      "companies": [
        {
          "name": "企業名",
          "business": "事業内容（50字程度）",
          "revenue": "年商XX億円",
          "employees": "従業員数XX名",
          "founded": "設立XXXX年",
          "location": "本社：XX都XX区",
          "vision": "企業ビジョン（30字程度）",
          "products": "主力製品・サービス（30字程度）",
          "culture": "企業文化（30字程度）",
          "recent_news": "最近のニュース・動向（30字程度）",
          "competitive_advantage": "競合優位性（30字程度）",
          "ceo_message": "CEO・代表メッセージ（30字程度）",
          "expansion_plan": "事業展開計画（30字程度）",
          "awards": "受賞歴・評価（30字程度）",
          "partnerships": "パートナーシップ・提携（30字程度）"
        }
      ]
    }
    ```

    要件：
    - 実在しない架空の企業名を使用
    - 多様な業界（IT、製造業、サービス業など）
    - リアルな規模感（従業員数100-5000名程度）
    - 現代的な日本企業として設定
    """
    response = call_gemini_api(GENERATOR_MODEL_NAME, prompt)
    data = parse_json_from_response(response)
    
    if isinstance(data, dict) and "companies" in data and len(data["companies"]) > 0:
        company_data = data["companies"][0]
        print("--- 企業情報の生成完了 ---")
        return company_data
    else:
        print("--- 企業情報の生成に失敗、または予期しない形式です ---")
        return {"error": "Failed to generate company profile correctly", "raw_data": data}


def generate_candidate_profiles(company_profile, num_candidates):
    """企業情報に基づき、複数の学生プロフィールを生成する"""
    print(f"--- {num_candidates}人の学生プロフィールの生成を開始 ---")
    prompt = f"""
    以下の企業情報に興味を持つ、日本の架空の就活生{num_candidates}人の情報を生成してください。

    [企業情報]
    {json.dumps(company_profile, ensure_ascii=False)}

    以下のJSON形式で出力してください：

    ```json
    {{
      "candidates": [
        {{
          "name": "日本人の氏名",
          "university": "大学名学部名",
          "gakuchika": "学生時代に力を入れたこと（100字程度の具体的なエピソード）",
          "interest": "興味のある分野・職種",
          "strength": "強み・能力",
          "preparation": "high",
          "mbti": "MBTIタイプ（例：INTJ、ESFJなど）"
        }}
      ]
    }}
    ```

    要件：
    - 多様な大学・学部（国公立・私立・地方大学含む）
    - 具体的で説得力のあるガクチカエピソード
    - 異なる強みや興味分野を持つ学生
    - リアルな日本の就活生として設定
    - `preparation`は、{num_candidates}人分を 'high', 'medium', 'low' に1人ずつ割り当ててください。
    """
    response = call_gemini_api(GENERATOR_MODEL_NAME, prompt)
    data = parse_json_from_response(response)
    
    if isinstance(data, dict) and "candidates" in data:
        candidates_data = data["candidates"]
        print("--- 学生プロフィールの生成完了 ---")
        return candidates_data
    else:
        print("--- 学生プロフィールの生成に失敗、または予期しない形式です ---")
        return {"error": "Failed to generate candidate profiles correctly", "raw_data": data}

# ==============================================================================
# 4. ローカルモデルの初期化
# ==============================================================================
def initialize_local_model():
    """Hugging Faceからローカルモデルを読み込み、GPUに配置する"""
    print(f"--- ローカルモデル ({LOCAL_MODEL_NAME}) の初期化を開始 ---")
    if not torch.cuda.is_available():
        print("警告: CUDAが利用できません。CPUでの実行は非常に遅くなります。")
        quantization_config = None
        torch_dtype = torch.float32
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        torch_dtype = torch.bfloat16
        print("CUDAを検出。4bit量子化を有効にしてモデルを読み込みます。")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, quantization_config=quantization_config, torch_dtype=torch_dtype, trust_remote_code=True)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
            
        print("--- ローカルモデルの初期化完了 ---")
        return model, tokenizer
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}\nHugging Face Hubへのログインが必要な場合があります (`huggingface-cli login`)。")
        return None, None

# ==============================================================================
# 5. LLMロール定義 (Classes)
# ==============================================================================

class InterviewerLLM:
    """面接官役のLLM (Gemini APIを使用)"""
    def __init__(self, company_profile):
        self.model_name = INTERVIEWER_MODEL_NAME
        self.company = company_profile

    def ask_question(self, conversation_history):
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history])
        prompt = f"""あなたは、{self.company.get('name', '私たちの会社')} という企業の採用面接官です。企業の詳細: {self.company.get('business', 'N/A')}
        以下の会話履歴を読み、学生の回答を分析し、次にするべき質問を考えてください。学生の企業理解度、論理性、熱意を測ることを目的とします。
        思考プロセスを内部で実行し、最終的な「質問」だけを返してください。
        ---
        会話履歴:
        {history_str if history_str else "（まだ会話はありません）"}
        ---
        次の質問:"""
        question = call_gemini_api(self.model_name, prompt)
        thought = f"履歴を分析し、質問 '{question[:30]}...' を生成しました。"
        return question, thought

    def evaluate_applicant(self, conversation_history, applicant_profile):
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history])
        prompt = f"""あなたは、{self.company.get('name', '私たちの会社')} のベテラン面接官です。以下の学生プロフィールと面接の会話履歴全体をレビューし、最終評価を行ってください。
        [企業情報]: {json.dumps(self.company, ensure_ascii=False)}
        [学生プロフィール]: {json.dumps(applicant_profile, ensure_ascii=False)}
        [面接の全会話履歴]:\n{history_str}
        [評価項目]: 一貫性, 具体性, 論理性, 熱意 (各1-5点)
        [出力形式]: 以下のキーを持つJSON形式で評価を出力してください。
        {{
          "overall_score": 総合評価 (5段階評価の浮動小数点数),
          "summary": "評価の要約 (100字程度)",
          "evaluation_details": {{"consistency":点数,"specificity":点数,"logicality":点数,"enthusiasm":点数}}
        }}"""
        response = call_gemini_api(self.model_name, prompt)
        return parse_json_from_response(response)

class CompanyKnowledgeManager:
    """学生が知っているべき企業情報を管理するクラス"""
    def __init__(self, full_company_profile):
        self.full_profile = full_company_profile
        self.all_keys = list(full_company_profile.keys())
        self.essential_keys = ["name", "business", "products", "vision"]

    def _punch_holes_in_string(self, text: str, percentage: float, placeholder: str = '_') -> str:
        """
        文字列を単語単位で穴抜きする。
        指定された割合の単語を '_' の連続に置換する。
        「」内の単語に対しては、穴抜き率を低減する。
        非単語文字（スペース、句読点など）は穴抜きしない。
        """
        if not isinstance(text, str) or not text:
            return text

        result_tokens = []
        
        segments = re.split(r'(「.*?」)', text)
        
        for segment in segments:
            is_quoted = segment.startswith('「') and segment.endswith('」') and len(segment) > 1
            
            effective_percentage = (percentage / 3.0) if is_quoted else percentage
            
            inner_text = segment[1:-1] if is_quoted else segment
            
            tokens = re.findall(r'[一-龯ぁ-んァ-ヶ々ーa-zA-Z0-9]+|\s+|.', inner_text)
            
            punched_tokens = []
            for token in tokens:
                if re.fullmatch(r'[一-龯ぁ-んァ-ヶ々ーa-zA-Z0-9]+', token) and random.random() < effective_percentage:
                    punched_tokens.append(placeholder * len(token))
                else:
                    punched_tokens.append(token)
            
            processed_segment = "".join(punched_tokens)
            if is_quoted:
                result_tokens.append('「' + processed_segment + '」')
            else:
                result_tokens.append(processed_segment)
        
        return "".join(result_tokens)

    def get_knowledge_for_level(self, level='high'):
        """準備レベルに応じて、フィルタリングされ、穴の開いた企業情報と知識のカバレッジ率を返す"""
        keys_to_keep_set = set()
        if level == 'high':
            keys_to_keep_set = set(self.all_keys)
        else:
            keys_to_keep_set.update(self.essential_keys)
            other_keys = [k for k in self.all_keys if k not in self.essential_keys]
            
            ratio = 0.0
            if level == 'medium':
                ratio = 1.0
            elif level == 'low':
                ratio = 1.0
            
            sample_size = int(len(other_keys) * ratio)
            if sample_size > 0:
                keys_to_keep_set.update(random.sample(other_keys, sample_size))
        
        hole_percentage = 0.0
        if level == 'medium':
            hole_percentage = 0.2
        elif level == 'low':
            hole_percentage = 0.4

        knowledge_dict = {}
        for key in self.all_keys:
            if key in keys_to_keep_set:
                value = self.full_profile[key]
                if level != 'high' and key not in self.essential_keys:
                    knowledge_dict[key] = self._punch_holes_in_string(value, hole_percentage)
                else:
                    knowledge_dict[key] = value
            else:
                knowledge_dict[key] = "" 

        coverage_percentage = int(len(keys_to_keep_set) / len(self.all_keys) * 100) if self.all_keys else 100
        coverage_str = f"{len(keys_to_keep_set)}/{len(self.all_keys)}項目 ({coverage_percentage}%)"
        
        return knowledge_dict, coverage_str


class InstructionPromptManager:
    """Llama 3.1用のプロンプトを生成するクラス"""
    def __init__(self):
        self.system_prompt = "あなたは優秀な日本語AIアシスタントです。指示に従って適切に回答してください。"

    @staticmethod
    def create_instruction_prompt(preparation_level):
        """回答方針作成メソッド"""
        if preparation_level == "high":
            return """
- 非常に高い志望度と熱意を必ず示してください。
- 他の就活生に負けない強い意欲を表現してください。
- 知っている具体的な情報は積極的に言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを強く表現してください。"""
        elif preparation_level == "medium":
            return """
- 高い志望度と熱意を必ず示してください。
- 他の就活生に負けない意欲を表現してください。
- 知っている具体的な情報は積極的に言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを表現してください。
- 企業情報の不足している部分(〇〇)は推測しながら話してください。
"""
        else: # low
            return """
- そこそこ高い志望度と熱意を必ず示してください。
- 他の就活生に負けない意欲をなるべく表現してください。
- 知っている具体的な情報は言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを表現してください。
- 企業情報の不足している部分(〇〇)は推測しながら話してください。
"""

    def _format_available_company_info(self, company_knowledge):
        return "\n".join([f"- {key}: {value}" for key, value in company_knowledge.items() if value])

    def _format_history(self, conversation_history):
        if not conversation_history:
            return "（まだ会話はありません）"
        return "\n".join([f"- 面接官: {turn['question']}\n- あなた: {turn['answer']}" for turn in conversation_history])

    def create_messages(self, candidate_profile, company_knowledge_tuple, conversation_history, new_question):
        company_knowledge, knowledge_coverage = company_knowledge_tuple
        preparation_level = candidate_profile.get('preparation', 'low')
        instruction_prompt = self.create_instruction_prompt(preparation_level)

        user_content = f"""
あなたは {candidate_profile.get("name", "名無しの候補者")} という日本の就活生です。この企業に絶対に入社したく、面接官に志望度の高さを強くアピールしたいと考えています。
企業研究を熱心に行いましたが、情報収集には限界があります。知っている情報は具体的に、知らない情報は前向きな推測や一般論で補って回答してください。

# あなたのプロフィール
- 氏名: {candidate_profile.get("name", "N/A")}
- 大学: {candidate_profile.get("university", "N/A")}
- 強み: {candidate_profile.get("strength", "N/A")}
- ガクチカ: {candidate_profile.get("gakuchika", "N/A")}
- MBTI: {candidate_profile.get("mbti", "N/A")}

# あなたが調べて得た企業情報（{knowledge_coverage}）
{self._format_available_company_info(company_knowledge)}

# 回答の重要な方針
{instruction_prompt}

# これまでの会話
{self._format_history(conversation_history)}

# 面接官からの質問
{new_question}

---
{candidate_profile.get("name", "名無しの候補者")} として、最高レベルの志望度と熱意を示しながら、150文字程度で自然な日本語で回答してください。
この企業への強い憧れと、絶対に入社したい気持ちを表現してください。
回答のみを出力し、説明や前置きは不要です。
"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        return messages

class LLamaInterviewResponseGenerator:
    """ローカルLLMからの応答生成を統括するクラス"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_manager = InstructionPromptManager()

    def generate(self, candidate_profile, company_knowledge_tuple, conversation_history, new_question):
        messages = self.prompt_manager.create_messages(
            candidate_profile, company_knowledge_tuple, conversation_history, new_question
        )
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        attention_mask = torch.ones_like(inputs).to(self.model.device)
        inputs = inputs.to(self.model.device)
        
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        response = outputs[0][inputs.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

# ==============================================================================
# 6. 実験実行コントローラー
# ==============================================================================
def run_experiment(local_model, local_tokenizer):
    """面接シミュレーション全体を実行する"""
    # --- 1. 動的情報生成 ---
    company_profile = generate_company_profile()
    if not isinstance(company_profile, dict) or "error" in company_profile:
        print("企業情報の生成に失敗したか、予期しない形式です。実験を中止します。")
        return

    candidate_profiles = generate_candidate_profiles(company_profile, NUM_CANDIDATES)
    if not isinstance(candidate_profiles, list) or (len(candidate_profiles) > 0 and "error" in candidate_profiles[0]):
        print("学生プロフィールの生成に失敗したか、予期しない形式です。実験を中止します。")
        if isinstance(candidate_profiles, dict) and 'raw_output' in candidate_profiles:
            print(f"RAW OUTPUT:\n{candidate_profiles['raw_output']}")
        return

    # --- 2. 各種マネージャーと候補者情報の初期化 ---
    interviewer = InterviewerLLM(company_profile)
    knowledge_manager = CompanyKnowledgeManager(company_profile)
    response_generator = LLamaInterviewResponseGenerator(local_model, local_tokenizer)
    
    # 各候補者の状態を管理するリストを作成
    candidate_states = []
    for profile in candidate_profiles:
        candidate_states.append({
            "profile": profile,
            "knowledge_tuple": knowledge_manager.get_knowledge_for_level(profile.get('preparation', 'low')),
            "conversation_log": []
        })

    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    common_question_index = 0

    # --- 3. 面接フローの実行 ---
    for round_num, question_type in enumerate(INTERVIEW_FLOW):
        print(f"\n{'='*80}\n--- 面接ラウンド {round_num + 1}/{len(INTERVIEW_FLOW)} ---\n{'='*80}")

        if question_type == 0: # 全体質問
            if common_question_index >= len(COMMON_QUESTIONS):
                print("警告: 共通質問が不足しています。このラウンドをスキップします。")
                continue
            
            question = COMMON_QUESTIONS[common_question_index]
            print(f"--- 全体質問: 「{question}」 ---")
            
            for i, state in enumerate(candidate_states):
                profile = state["profile"]
                print(f"\n -> 候補者 {i+1}: {profile.get('name', 'N/A')} へ質問")
                
                print("  学生 (Local) が応答を生成中...")
                answer = response_generator.generate(
                    profile, state["knowledge_tuple"], state["conversation_log"], question
                )
                print(f"  学生 (Local): {answer}")
                state["conversation_log"].append({"turn": round_num + 1, "question": question, "answer": answer})
            
            common_question_index += 1

        elif question_type == 1: # 個別質問
            print("--- 個別質問フェーズ ---")
            for i, state in enumerate(candidate_states):
                profile = state["profile"]
                print(f"\n -> 候補者 {i+1}: {profile.get('name', 'N/A')} へ質問")
                
                question, _ = interviewer.ask_question(state["conversation_log"])
                print(f"  面接官 (API): {question}")

                print("  学生 (Local) が応答を生成中...")
                answer = response_generator.generate(
                    profile, state["knowledge_tuple"], state["conversation_log"], question
                )
                print(f"  学生 (Local): {answer}")
                state["conversation_log"].append({"turn": round_num + 1, "question": question, "answer": answer})

    # --- 4. 最終評価 ---
    print(f"\n{'='*80}\n--- 最終評価フェーズ ---\n{'='*80}")
    all_results = []
    for i, state in enumerate(candidate_states):
        print(f"候補者 {i+1}: {state['profile'].get('name', 'N/A')} の評価を生成中...")
        final_evaluation = interviewer.evaluate_applicant(state["conversation_log"], state["profile"])
        
        candidate_result = {
            "candidate_info": state["profile"],
            "candidate_knowledge": state["knowledge_tuple"][0], # 辞書のみ保存
            "final_evaluation": final_evaluation,
            "conversation_log": state["conversation_log"]
        }
        all_results.append(candidate_result)

    # --- 5. 全結果の保存 ---
    final_output = {
        "experiment_info": {
            "interviewer_model": interviewer.model_name,
            "applicant_model": LOCAL_MODEL_NAME,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "company_profile": company_profile,
        "interview_results": all_results
    }

    if not os.path.exists('results'): os.makedirs('results')
    filename = f"results/experiment_{timestamp_str}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"\n{'='*80}\n--- 全ての実験が完了しました ---")
    print(f"結果を {filename} に保存しました。")

# ==============================================================================
# 7. メイン処理 (Main Execution)
# ==============================================================================
if __name__ == "__main__":
    local_model, local_tokenizer = initialize_local_model()

    if local_model and local_tokenizer:
        run_experiment(local_model, local_tokenizer)
    else:
        print("ローカルモデルの初期化に失敗したため、実験を中止します。")
