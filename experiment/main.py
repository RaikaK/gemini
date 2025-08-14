# main.py

import os
import json
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 分割したファイルから必要なモジュールをインポート
import config
import data_generators
import student as student_module
import interv as interviewer_module
import benchmark

def initialize_local_model():
    """Hugging Faceからローカルモデルを読み込み、GPUに配置する"""
    print(f"--- ローカルモデル ({config.LOCAL_MODEL_NAME}) の初期化を開始 ---")
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
            
        print("--- ローカルモデルの初期化完了 ---")
        return model, tokenizer
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}\nHugging Face Hubへのログインが必要な場合があります (`huggingface-cli login`)。")
        return None, None

def run_experiment(local_model, local_tokenizer):
    """面接シミュレーション全体を実行する"""
    # --- 1. 動的情報生成 ---
    company_profile = data_generators.generate_company_profile()
    if not isinstance(company_profile, dict) or "error" in company_profile:
        print("企業情報の生成に失敗したか、予期しない形式です。実験を中止します。")
        return

    candidate_profiles = data_generators.generate_candidate_profiles(company_profile, config.NUM_CANDIDATES)
    if not isinstance(candidate_profiles, list) or (len(candidate_profiles) > 0 and "error" in candidate_profiles[0]):
        print("学生プロフィールの生成に失敗したか、予期しない形式です。実験を中止します。")
        if isinstance(candidate_profiles, dict) and 'raw_output' in candidate_profiles:
            print(f"RAW OUTPUT:\n{candidate_profiles['raw_output']}")
        return

    # --- 2. 各種マネージャーと候補者情報の初期化 ---
    interviewer = interviewer_module.InterviewerLLM(company_profile)
    knowledge_manager = student_module.CompanyKnowledgeManager(company_profile)
    response_generator = interviewer_module.LLamaInterviewResponseGenerator(local_model, local_tokenizer)
    
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
    for round_num, question_type in enumerate(config.INTERVIEW_FLOW):
        print(f"\n{'='*80}\n--- 面接ラウンド {round_num + 1}/{len(config.INTERVIEW_FLOW)} ---\n{'='*80}")

        if question_type == 0: # 全体質問
            if common_question_index >= len(config.COMMON_QUESTIONS):
                print("警告: 共通質問が不足しています。このラウンドをスキップします。")
                continue
            
            question = config.COMMON_QUESTIONS[common_question_index]
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

    # --- 4. 最終評価 & ベンチマーク ---
    print(f"\n{'='*80}\n--- 最終評価 & ベンチマークフェーズ ---\n{'='*80}")
    all_results = []
    benchmark_evaluator = benchmark.BenchmarkEvaluator(company_profile)

    for i, state in enumerate(candidate_states):
        print(f"\n候補者 {i+1}: {state['profile'].get('name', 'N/A')} の評価を生成中...")
        
        # a) 面接官による総合評価
        interviewer_evaluation = interviewer.evaluate_applicant(state["conversation_log"], state["profile"])
        
        candidate_knowledge = state["knowledge_tuple"][0]
        
        # b) 学生の補完能力ベンチマーク
        student_completion_benchmark = benchmark_evaluator.evaluate_student_completion(
            candidate_knowledge,
            state["conversation_log"]
        )
        
        # c) 面接官の質問品質ベンチマーク
        interviewer_probing_benchmark = benchmark_evaluator.evaluate_interviewer_performance(
            candidate_knowledge,
            state["conversation_log"],
            config.INTERVIEW_FLOW
        )
        
        # ★ 新設: サマリースコアの計算と表示
        # ----------------------------------------------------------------------
        # 学生の補完能力スコア
        completion_scores = [
            turn['evaluation']['plausible_completion']['score']
            for turn in student_completion_benchmark.get('turn_by_turn_evaluation', [])
            if 'plausible_completion' in turn.get('evaluation', {}) and 'score' in turn['evaluation']['plausible_completion']
        ]
        avg_completion_score = sum(completion_scores) / len(completion_scores) if completion_scores else 0.0

        # 面接官の質問品質スコア
        probing_scores = [
            turn['evaluation']['targeted_probing_score']['score']
            for turn in interviewer_probing_benchmark.get('individual_question_evaluation', [])
            if 'targeted_probing_score' in turn.get('evaluation', {}) and 'score' in turn['evaluation']['targeted_probing_score']
        ]
        avg_probing_score = sum(probing_scores) / len(probing_scores) if probing_scores else 0.0

        # 面接官の最終評価スコア
        interviewer_overall_score = interviewer_evaluation.get('overall_score', 0.0)

        print(f"--- スコアサマリー: {state['profile'].get('name', 'N/A')} ---")
        print(f"  - 面接官の最終評価: {interviewer_overall_score:.2f} / 5.0")
        print(f"  - 学生の補完能力 (平均): {avg_completion_score:.2f} / 5.0")
        print(f"  - 面接官の質問品質 (平均): {avg_probing_score:.2f} / 5.0")
        # ----------------------------------------------------------------------

        candidate_result = {
            "candidate_info": state["profile"],
            "candidate_knowledge": candidate_knowledge,
            "interviewer_evaluation": interviewer_evaluation,
            "student_completion_benchmark": student_completion_benchmark,
            "interviewer_probing_benchmark": interviewer_probing_benchmark,
            "conversation_log": state["conversation_log"]
        }
        all_results.append(candidate_result)


    # --- 5. 全結果の保存 ---
    final_output = {
        "experiment_info": {
            "interviewer_model": config.INTERVIEWER_MODEL_NAME,
            "applicant_model": config.LOCAL_MODEL_NAME,
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
