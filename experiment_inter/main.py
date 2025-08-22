# main.py

import os
import json
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 分割したファイルから必要なモジュールをインポート
import config
import data_generators
# student.pyからはCompanyKnowledgeManagerと新しいGPTApplicantをインポート
from student import CompanyKnowledgeManager, GPTApplicant
# interv.pyからは新しいLocalInterviewerLLMをインポート
from interv import LocalInterviewerLLM

def initialize_local_model():
    """Hugging Faceからローカルモデルを読み込み、GPUに配置する (今回は面接官役)"""
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

def run_experiment(local_interviewer_model, local_interviewer_tokenizer):
    """面接シミュレーション全体を実行する (学生: API, 面接官: Local)"""
    # --- 1. 動的情報生成 ---
    company_profile = data_generators.generate_company_profile()
    if not isinstance(company_profile, dict) or "error" in company_profile:
        print("企業情報の生成に失敗。実験を中止します。")
        return

    candidate_profiles = data_generators.generate_candidate_profiles(company_profile, config.NUM_CANDIDATES)
    if not isinstance(candidate_profiles, list) or not candidate_profiles:
        print("学生プロフィールの生成に失敗。実験を中止します。")
        return

    # --- 2. 各種マネージャーと候補者情報の初期化 (役割交代) ---
    interviewer = LocalInterviewerLLM(company_profile, local_interviewer_model, local_interviewer_tokenizer)
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
    # 質問済みの全体質問を保持するリストを作成
    asked_common_questions = []

    # --- 3. 面接フローの実行 ---
    for round_num, question_type in enumerate(config.INTERVIEW_FLOW):
        print(f"\n{'='*80}\n--- 面接ラウンド {round_num + 1}/{len(config.INTERVIEW_FLOW)} ---\n{'='*80}")

        if question_type == 0: # 全体質問
            print("--- 全体質問フェーズ ---")
            print("  面接官 (Local) が全体質問を生成中...")
            question, _ = interviewer.ask_common_question(asked_common_questions)
            asked_common_questions.append(question) # 生成した質問をリストに追加
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
                print("  面接官 (Local) が質問を生成中...")
                question, _ = interviewer.ask_question(state["conversation_log"])
                print(f"  面接官 (Local): {question}")
                print("  学生 (API) が応答を生成中...")
                answer = applicant.generate(
                    state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                )
                print(f"  学生 (API): {answer}")
                state["conversation_log"].append({"turn": round_num + 1, "question": question, "answer": answer})

    # --- 4. 最終評価 (新しい3つの全体評価タスク) ---
    print(f"\n{'='*80}\n--- 最終評価フェーズ ---\n{'='*80}")
    
    best_candidate_eval = interviewer.select_best_candidate(candidate_states)
    print("\n[評価1: 最優秀候補者]")
    print(best_candidate_eval)
    
    ranking_eval = interviewer.rank_candidates(candidate_states)
    print("\n[評価2: 候補者ランキング]")
    print(ranking_eval)

    knowledge_gap_eval = interviewer.detect_knowledge_gaps(candidate_states)
    print("\n[評価3: 知識欠損の分析]")
    print(json.dumps(knowledge_gap_eval, ensure_ascii=False, indent=2)) # 整形して表示
    
    # --- 5. 全結果の保存 ---
    final_output = {
        "experiment_info": {
            "interviewer_model": config.LOCAL_MODEL_NAME,
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
            "best_candidate": best_candidate_eval,
            "ranking": ranking_eval,
            "knowledge_gaps": knowledge_gap_eval
        }
    }

    if not os.path.exists('results'): os.makedirs('results')
    filename = f"results/experiment_reversed_{timestamp_str}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"\n{'='*80}\n--- 全ての実験が完了しました ---")
    print(f"結果を {filename} に保存しました。")


if __name__ == "__main__":
    local_model, local_tokenizer = initialize_local_model()

    if local_model and local_tokenizer:
        run_experiment(local_model, local_tokenizer)
    else:
        print("ローカルモデルの初期化に失敗したため、実験を中止します。")