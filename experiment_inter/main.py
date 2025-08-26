# main.py

import os
import json
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import config
import data_generators
from student import CompanyKnowledgeManager, GPTApplicant
from interv import Interviewer # 統合されたInterviewerクラスをインポート

def initialize_local_model():
    """Hugging Faceからローカルモデルを読み込み、GPUに配置する"""
    print(f"--- 面接官役のローカルモデル ({config.LOCAL_MODEL_NAME}) の初期化を開始 ---")
    if not torch.cuda.is_available():
        print("警告: CUDAが利用できません。CPUでの実行は非常に遅くなります。")
        quantization_config = None
        torch_dtype = torch.float32
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        torch_dtype = torch.bfloat16
        print("CUDAを検出。4bit量子化を有効にしてモデルを読み込みます。")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(config.LOCAL_MODEL_NAME, quantization_config=quantization_config, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
            
        print("--- 面接官役のローカルモデルの初期化完了 ---")
        return model, tokenizer
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}")
        return None, None

def run_experiment(local_interviewer_model=None, local_interviewer_tokenizer=None, set_index=None):
    """面接シミュレーション全体を実行する"""
    # --- 1. db.jsonからデータ読み込み ---
    company_profile, candidate_profiles = data_generators.load_company_and_candidates_from_db(set_index)
    if company_profile is None or candidate_profiles is None:
        print("db.jsonからのデータ読み込みに失敗。実験を中止します。")
        return

    if len(candidate_profiles) < config.NUM_CANDIDATES:
        print(f"警告: 読み込まれた学生数({len(candidate_profiles)})が設定値({config.NUM_CANDIDATES})より少ないため、利用可能な学生を使用します。")
        candidate_profiles = candidate_profiles[:config.NUM_CANDIDATES]

    # --- 2. 面接官と候補者の初期化 ---
    model_type = config.INTERVIEWER_MODEL_TYPE
    print(f"--- 面接官タイプ: {model_type} ---")
    try:
        if model_type == 'local':
            interviewer = Interviewer(
                company_profile=company_profile,
                model_type='local',
                model=local_interviewer_model,
                tokenizer=local_interviewer_tokenizer
            )
        elif model_type == 'api':
            interviewer = Interviewer(
                company_profile=company_profile,
                model_type='api'
            )
        else:
            print(f"エラー: config.pyのINTERVIEWER_MODEL_TYPEに無効な値 '{model_type}' が設定されています。")
            return
    except ValueError as e:
        print(f"エラー: {e}")
        return

    knowledge_manager = CompanyKnowledgeManager(company_profile)
    applicant = GPTApplicant(config.APPLICANT_API_MODEL)
    
    candidate_states = []
    for profile in candidate_profiles:
        candidate_states.append({
            "profile": profile,
            "knowledge_tuple": knowledge_manager.get_knowledge_for_level(profile.get('preparation', 'low')),
            "conversation_log": []
        })

    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    asked_common_questions = []

    # --- 3. 面接フローの実行 ---
    for round_num, question_type in enumerate(config.INTERVIEW_FLOW):
        print(f"\n{'='*80}\n--- 面接ラウンド {round_num + 1}/{len(config.INTERVIEW_FLOW)} ---\n{'='*80}")

        if question_type == 0: # 全体質問
            print("--- 全体質問フェーズ ---")
            print(f"  面接官 ({model_type}) が全体質問を生成中...")
            question, _ = interviewer.ask_common_question(asked_common_questions)
            asked_common_questions.append(question)
            print(f"--- 生成された全体質問: 「{question}」 ---")
            
            for i, state in enumerate(candidate_states):
                print(f"\n -> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                print("  学生 (API) が応答を生成中...")
                answer = applicant.generate(
                    state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                )
                print(f"  学生 (API): {answer}")
                state["conversation_log"].append({"turn": round_num + 1, "question": question, "answer": answer})

        elif question_type == 1: # 個別質問
            print("--- 個別質問フェーズ ---")
            for i, state in enumerate(candidate_states):
                print(f"\n -> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                print(f"  面接官 ({model_type}) が質問を生成中...")
                question, _ = interviewer.ask_question(state["conversation_log"])
                print(f"  面接官 ({model_type}): {question}")
                print("  学生 (API) が応答を生成中...")
                answer = applicant.generate(
                    state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                )
                print(f"  学生 (API): {answer}")
                state["conversation_log"].append({"turn": round_num + 1, "question": question, "answer": answer})

    # --- 4. 最終評価 ---
    print(f"\n{'='*80}\n--- 最終評価フェーズ ---\n{'='*80}")
    
    least_motivated_eval = interviewer.select_least_motivated_candidate(candidate_states)
    print("\n[評価1: 最も志望度が低い候補者]")
    print(least_motivated_eval)
    
    ranking_eval = interviewer.rank_candidates_by_motivation(candidate_states)
    print("\n[評価2: 候補者ランキング（志望度が低い順）]")
    print(ranking_eval)

    knowledge_gap_eval = interviewer.detect_knowledge_gaps(candidate_states)
    print("\n[評価3: 知識欠損の分析]")
    print(json.dumps(knowledge_gap_eval, ensure_ascii=False, indent=2))
    
    # --- 5. 全結果の保存 ---
    final_output = {
        "experiment_info": {
            "interviewer_model": config.LOCAL_MODEL_NAME if model_type == 'local' else config.INTERVIEWER_API_MODEL,
            "interviewer_type": model_type,
            "applicant_model": config.APPLICANT_API_MODEL,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "company_profile": company_profile,
        "interview_transcripts": [
            {
                "candidate_info": s["profile"],
                "possessed_company_knowledge": s["knowledge_tuple"][0],
                "knowledge_coverage_info": s["knowledge_tuple"][1],
                "conversation_log": s["conversation_log"]
            }
            for s in candidate_states
        ],
        "final_evaluations": {
            "least_motivated_candidate": least_motivated_eval,
            "motivation_ranking": ranking_eval,
            "knowledge_gaps": knowledge_gap_eval
        }
    }

    if not os.path.exists('results'): os.makedirs('results')
    filename = f"results/experiment_results_{timestamp_str}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"\n{'='*80}\n--- 全ての実験が完了しました ---")
    print(f"結果を {filename} に保存しました。")


if __name__ == "__main__":
    model_type = config.INTERVIEWER_MODEL_TYPE
    
    # セットインデックスを指定する場合はここで設定（Noneの場合はランダム選択）
    set_index = None  # 例: set_index = 0 で最初のセットを選択
    
    if model_type == 'local':
        local_model, local_tokenizer = initialize_local_model()
        if local_model and local_tokenizer:
            run_experiment(local_model, local_tokenizer, set_index)
        else:
            print("ローカルモデルの初期化に失敗したため、実験を中止します。")
    elif model_type == 'api':
        run_experiment(set_index=set_index)
    else:
        print(f"エラー: config.pyのINTERVIEWER_MODEL_TYPEに無効な値 '{model_type}' が設定されています。")