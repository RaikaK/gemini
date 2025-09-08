from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import datetime
import threading
import queue
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 既存のモジュールをインポート
import config
import data_generators
from student import CompanyKnowledgeManager, GPTApplicant
from interv import Interviewer

app = Flask(__name__)

# グローバル変数
experiment_status = {
    'is_running': False,
    'progress': 0,
    'current_step': '',
    'logs': [],
    'results': None,
    'current_simulation': 0,
    'total_simulations': 0,
    'simulation_results': []
}

experiment_queue = queue.Queue()

# 絶対パスでresultsディレクトリを指定
RESULTS_DIR = Path(__file__).parent / 'results'

def log_message(message):
    """ログメッセージを追加"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    experiment_status['logs'].append(log_entry)
    if len(experiment_status['logs']) > 100:  # ログを100件まで保持
        experiment_status['logs'] = experiment_status['logs'][-100:]

def update_progress(progress, step):
    """進捗状況を更新"""
    experiment_status['progress'] = progress
    experiment_status['current_step'] = step

def calculate_accuracy_metrics(candidate_states, least_motivated_eval, ranking_eval=None, knowledge_gaps_eval=None):
    """既存のinterv.pyの評価手法に基づく精度指標を計算"""
    try:
        # 真の志望度が最も低い候補者を特定
        true_least_motivated = None
        min_motivation = float('inf')
        
        for i, state in enumerate(candidate_states):
            profile = state['profile']
            # 準備レベルから志望度を推定（低いほど志望度が低い）
            preparation_levels = {'low': 1, 'medium': 2, 'high': 3}
            motivation_score = preparation_levels.get(profile.get('preparation', 'low'), 1)
            
            if motivation_score < min_motivation:
                min_motivation = motivation_score
                true_least_motivated = {
                    'index': i,
                    'name': profile.get('name', f'Candidate_{i+1}'),
                    'preparation': profile.get('preparation', 'low')
                }
        
        # 評価1: 最も志望度が低い候補者の選定精度
        predicted_least_motivated = ""
        if isinstance(least_motivated_eval, str):
            import re
            match = re.search(r'最も志望度が低い候補者:\s*([^\n]+)', least_motivated_eval)
            if match:
                predicted_least_motivated = match.group(1).strip()
            else:
                candidate_names = [state['profile'].get('name', f'Candidate_{i+1}') for i, state in enumerate(candidate_states)]
                for name in candidate_names:
                    if name in least_motivated_eval:
                        predicted_least_motivated = name
                        break
                if not predicted_least_motivated:
                    predicted_least_motivated = "不明"
        else:
            predicted_least_motivated = str(least_motivated_eval)
        
        is_correct = (true_least_motivated['name'] == predicted_least_motivated)
        
        # 評価1の精度指標
        y_true = [1 if i == true_least_motivated['index'] else 0 for i in range(len(candidate_states))]
        y_pred = [1 if candidate_states[i]['profile'].get('name', f'Candidate_{i+1}') == predicted_least_motivated else 0 
                 for i in range(len(candidate_states))]
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 評価2: ランキング精度（オプション）
        ranking_accuracy = None
        if ranking_eval:
            ranking_accuracy = calculate_ranking_accuracy(candidate_states, ranking_eval)
        
        # 評価3: 知識欠損検出精度（オプション）
        knowledge_gaps_metrics = None
        knowledge_gaps_detailed = None
        if knowledge_gaps_eval and isinstance(knowledge_gaps_eval, dict):
            knowledge_gaps_metrics = knowledge_gaps_eval.get('quantitative_performance_metrics', {})
            knowledge_gaps_detailed = {
                'llm_qualitative_analysis': knowledge_gaps_eval.get('llm_qualitative_analysis', ''),
                'quantitative_performance_metrics': knowledge_gaps_eval.get('quantitative_performance_metrics', {})
            }
        
        return {
            'is_correct': is_correct,
            'true_least_motivated': true_least_motivated,
            'predicted_least_motivated': predicted_least_motivated,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'ranking_accuracy': ranking_accuracy,
            'knowledge_gaps_metrics': knowledge_gaps_metrics,
            'knowledge_gaps_detailed': knowledge_gaps_detailed
        }
    except Exception as e:
        log_message(f"精度指標の計算中にエラーが発生しました: {e}")
        return None

def calculate_ranking_accuracy(candidate_states, ranking_eval):
    """ランキング評価の精度指標を計算"""
    try:
        # 真の志望度ランキングを作成（低い順）
        true_ranking = []
        for i, state in enumerate(candidate_states):
            profile = state['profile']
            preparation = profile.get('preparation', 'low')
            preparation_levels = {'low': 1, 'medium': 2, 'high': 3}
            motivation_score = preparation_levels.get(preparation, 1)
            true_ranking.append({
                'name': profile.get('name', f'Candidate_{i+1}'),
                'score': motivation_score,
                'preparation': preparation
            })
        
        # 真のランキングをスコア順にソート（低い順）
        true_ranking.sort(key=lambda x: x['score'])
        
        # 予測ランキングを抽出
        predicted_ranking = []
        if isinstance(ranking_eval, str):
            import re
            # "1位: [氏名]" のパターンを探す
            for i in range(1, len(candidate_states) + 1):
                pattern = rf"{i}位:\s*([^\s(]+)"
                match = re.search(pattern, ranking_eval)
                if match:
                    predicted_ranking.append(match.group(1).strip())
                else:
                    predicted_ranking.append("不明")
        else:
            predicted_ranking = ["不明"] * len(candidate_states)
        
        # ランキングの一致度を計算
        true_names = [item['name'] for item in true_ranking]
        correct_positions = sum(1 for true, pred in zip(true_names, predicted_ranking) if true == pred)
        ranking_accuracy = correct_positions / len(true_names) if true_names else 0
        
        return {
            'accuracy': ranking_accuracy,
            'true_ranking': true_ranking,
            'predicted_ranking': predicted_ranking,
            'correct_positions': correct_positions,
            'total_positions': len(true_names)
        }
        
    except Exception as e:
        log_message(f"ランキング精度の計算中にエラーが発生しました: {e}")
        return None

def initialize_local_model(model_name=None):
    """Hugging Faceからローカルモデルを読み込み、GPUに配置する"""
    if model_name is None:
        model_name = config.LOCAL_MODEL_NAME
    
    log_message(f"--- 面接官役のローカルモデル ({model_name}) の初期化を開始 ---")
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    if not torch.cuda.is_available():
        log_message("警告: CUDAが利用できません。CPUでの実行は非常に遅くなります。")
        quantization_config = None
        torch_dtype = torch.float32
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        torch_dtype = torch.bfloat16
        log_message("CUDAを検出。4bit量子化を有効にしてモデルを読み込みます。")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
            
        log_message("--- 面接官役のローカルモデルの初期化完了 ---")
        return model, tokenizer
    except Exception as e:
        log_message(f"モデルの初期化中にエラーが発生しました: {e}")
        return None, None

def run_single_experiment(local_interviewer_model=None, local_interviewer_tokenizer=None, set_index=None, simulation_num=1, interview_flow=None, use_dynamic_flow=False, interviewer_model_type=None, interviewer_model_name=None):
    """単一の面接シミュレーション実行"""
    log_message(f"=== シミュレーション {simulation_num} 開始 ===")
    
    # 面接フローが指定されていない場合はデフォルトを使用
    if interview_flow is None:
        interview_flow = config.INTERVIEW_FLOW
    
    # モデル設定が指定されていない場合はconfigから取得
    if interviewer_model_type is None:
        interviewer_model_type = config.INTERVIEWER_MODEL_TYPE
    if interviewer_model_name is None:
        if interviewer_model_type == 'api':
            interviewer_model_name = config.INTERVIEWER_API_MODEL
        else:
            interviewer_model_name = config.LOCAL_MODEL_NAME
    
    # --- 1. db.jsonからデータ読み込み ---
    company_profile, candidate_profiles, actual_set_index = data_generators.load_company_and_candidates_from_db(set_index)
    if company_profile is None or candidate_profiles is None:
        log_message("db.jsonからのデータ読み込みに失敗。実験を中止します。")
        return None

    if len(candidate_profiles) < config.NUM_CANDIDATES:
        log_message(f"警告: 読み込まれた学生数({len(candidate_profiles)})が設定値({config.NUM_CANDIDATES})より少ないため、利用可能な学生を使用します。")
        candidate_profiles = candidate_profiles[:config.NUM_CANDIDATES]

    # --- 2. 面接官と候補者の初期化 ---
    log_message(f"--- 面接官タイプ: {interviewer_model_type} ({interviewer_model_name}) ---")
    
    try:
        if interviewer_model_type == 'local':
            # ローカルモデルが事前に初期化されていない場合は新しく初期化
            if local_interviewer_model is None or local_interviewer_tokenizer is None:
                log_message(f"ローカルモデル {interviewer_model_name} を新しく初期化します...")
                local_interviewer_model, local_interviewer_tokenizer = initialize_local_model(interviewer_model_name)
                if local_interviewer_model is None or local_interviewer_tokenizer is None:
                    log_message("ローカルモデルの初期化に失敗しました。")
                    return None
            
            interviewer = Interviewer(
                company_profile=company_profile,
                model_type='local',
                model=local_interviewer_model,
                tokenizer=local_interviewer_tokenizer
            )
        elif interviewer_model_type == 'api':
            # APIモデル名を動的に設定
            original_api_model = config.INTERVIEWER_API_MODEL
            config.INTERVIEWER_API_MODEL = interviewer_model_name
            try:
                interviewer = Interviewer(
                    company_profile=company_profile,
                    model_type='api'
                )
            finally:
                # 元の設定を復元
                config.INTERVIEWER_API_MODEL = original_api_model
        else:
            log_message(f"エラー: 無効な面接官モデルタイプ '{interviewer_model_type}' が指定されています。")
            return None
    except ValueError as e:
        log_message(f"エラー: {e}")
        return None

    knowledge_manager = CompanyKnowledgeManager(company_profile)
    applicant = GPTApplicant(config.APPLICANT_API_MODEL)
    
    candidate_states = []
    for profile in candidate_profiles:
        candidate_states.append({
            "profile": profile,
            "knowledge_tuple": knowledge_manager.get_knowledge_for_level(profile.get('preparation', 'low')),
            "conversation_log": []
        })

    asked_common_questions = []

    # --- 3. 面接フローの実行 ---
    actual_interview_flow = interview_flow  # デフォルトは指定されたフロー
    
    if use_dynamic_flow or config.USE_INTELLIGENT_DYNAMIC_FLOW:
        log_message("--- 智的動的面接フローを開始 ---")
        total_rounds, actual_interview_flow = interviewer.conduct_dynamic_interview(candidate_states, applicant, max_rounds=config.MAX_DYNAMIC_ROUNDS)
        log_message(f"--- 実際に実行された面接フロー: {actual_interview_flow} ---")
        
    else:
        # 従来の固定面接フロー
        total_rounds = len(interview_flow)
        for round_num, question_type in enumerate(interview_flow):
            log_message(f"--- 面接ラウンド {round_num + 1}/{total_rounds} ---")

            if question_type == 0: # 全体質問
                log_message("--- 全体質問フェーズ ---")
                question, _ = interviewer.ask_common_question(asked_common_questions)
                asked_common_questions.append(question)
                log_message(f"--- 生成された全体質問: 「{question}」 ---")
                
                for i, state in enumerate(candidate_states):
                    log_message(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                    answer, token_info = applicant.generate(
                        state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                    )
                    log_message(f"学生 (API): {answer}")
                    log_message(f"Token数: {token_info['total_tokens']} (プロンプト: {token_info['prompt_tokens']}, 回答: {token_info['completion_tokens']})")
                    state["conversation_log"].append({
                        "turn": round_num + 1, 
                        "question": question, 
                        "answer": answer,
                        "token_info": token_info
                    })

            elif question_type == 1: # 個別質問
                log_message("--- 個別質問フェーズ ---")
                for i, state in enumerate(candidate_states):
                    log_message(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                    question, _ = interviewer.ask_question(state["conversation_log"])
                    log_message(f"面接官 ({interviewer_model_type}): {question}")
                    answer, token_info = applicant.generate(
                        state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                    )
                    log_message(f"学生 (API): {answer}")
                    log_message(f"Token数: {token_info['total_tokens']} (プロンプト: {token_info['prompt_tokens']}, 回答: {token_info['completion_tokens']})")
                    state["conversation_log"].append({
                        "turn": round_num + 1, 
                        "question": question, 
                        "answer": answer,
                        "token_info": token_info
                    })

    # --- 4. 最終評価 ---
    log_message("--- 最終評価フェーズ ---")
    
    least_motivated_eval = interviewer.select_least_motivated_candidate(candidate_states)
    ranking_eval = interviewer.rank_candidates_by_motivation(candidate_states)
    knowledge_gap_eval = interviewer.detect_knowledge_gaps(candidate_states)
    
    # --- 5. 精度指標の計算 ---
    accuracy_metrics = calculate_accuracy_metrics(candidate_states, least_motivated_eval, ranking_eval, knowledge_gap_eval)
    
    # --- 6. 結果の整理 ---
    result = {
        "simulation_num": simulation_num,
        "experiment_info": {
            "dataset_index": actual_set_index,
            "dataset_name": f"Dataset_{actual_set_index + 1}",
            "interviewer_type": interviewer_model_type,
            "interviewer_model_name": interviewer_model_name,
            "interview_flow": actual_interview_flow,
            "use_dynamic_flow": use_dynamic_flow or config.USE_INTELLIGENT_DYNAMIC_FLOW,
            "total_rounds": total_rounds,
            "timestamp": datetime.datetime.now().isoformat(),
            "set_index": set_index  # 元のリクエスト値（Noneの場合はランダム）
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
        },
        "knowledge_gaps_detailed": accuracy_metrics.get('knowledge_gaps_detailed') if accuracy_metrics else None,
        "accuracy_metrics": accuracy_metrics
    }
    
    log_message(f"=== シミュレーション {simulation_num} 完了 ===")
    if accuracy_metrics:
        log_message(f"評価1 - 正解: {'✓' if accuracy_metrics['is_correct'] else '✗'}")
        log_message(f"評価1 - F1スコア: {accuracy_metrics['f1_score']:.3f}")
        log_message(f"評価1 - 正解率: {accuracy_metrics['accuracy']:.3f}")
        if accuracy_metrics.get('ranking_accuracy'):
            log_message(f"評価2 - ランキング正解率: {accuracy_metrics['ranking_accuracy']['accuracy']:.3f}")
        if accuracy_metrics.get('knowledge_gaps_metrics'):
            kg_metrics = accuracy_metrics['knowledge_gaps_metrics']
            log_message(f"評価3 - 知識欠損検出:")
            log_message(f"  - 検出された知識欠損数: {kg_metrics.get('detected_gaps_count', 'N/A')}")
            log_message(f"  - 真の知識欠損数: {kg_metrics.get('true_gaps_count', 'N/A')}")
            log_message(f"  - 検出精度: {kg_metrics.get('detection_accuracy', 'N/A')}")
            log_message(f"  - 検出F1スコア: {kg_metrics.get('detection_f1_score', 'N/A')}")
            log_message(f"  - 検出された欠損詳細: {kg_metrics.get('detected_gaps_details', 'N/A')}")
        else:
            log_message(f"評価3 - 知識欠損検出: データなし")
    
    return result

def run_experiment_web(local_interviewer_model=None, local_interviewer_tokenizer=None, set_index=None, num_simulations=1, interview_flow=None, use_dynamic_flow=False, interviewer_model_type=None, interviewer_model_name=None):
    """Web用の面接シミュレーション実行関数（複数回対応）"""
    try:
        experiment_status['is_running'] = True
        experiment_status['logs'] = []
        experiment_status['results'] = None
        experiment_status['simulation_results'] = []
        experiment_status['current_simulation'] = 0
        experiment_status['total_simulations'] = num_simulations
        
        # モデル設定が指定されていない場合はconfigから取得
        if interviewer_model_type is None:
            interviewer_model_type = config.INTERVIEWER_MODEL_TYPE
        if interviewer_model_name is None:
            if interviewer_model_type == 'api':
                interviewer_model_name = config.INTERVIEWER_API_MODEL
            else:
                interviewer_model_name = config.LOCAL_MODEL_NAME
        
        update_progress(0, f"{num_simulations}回のシミュレーションを開始しています...")
        log_message(f"面接官モデル設定: {interviewer_model_type} ({interviewer_model_name})")
        
        # 結果保存用のディレクトリ作成
        if not RESULTS_DIR.exists(): 
            RESULTS_DIR.mkdir(parents=True)
        
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        all_results = []
        
        for sim_num in range(1, num_simulations + 1):
            experiment_status['current_simulation'] = sim_num
            progress = (sim_num - 1) / num_simulations * 100
            update_progress(int(progress), f"シミュレーション {sim_num}/{num_simulations} を実行中...")
            
            # 単一実験の実行
            result = run_single_experiment(
                local_interviewer_model, local_interviewer_tokenizer, set_index, sim_num, 
                interview_flow, use_dynamic_flow, interviewer_model_type, interviewer_model_name
            )
            
            if result:
                all_results.append(result)
                experiment_status['simulation_results'].append(result)
                
                # 個別結果の保存
                individual_filename = f"simulation_{sim_num}_results_{timestamp_str}.json"
                with open(RESULTS_DIR / individual_filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
        
        # 全体結果の集計
        if all_results:
            # 精度指標の集計
            accuracy_scores = [r['accuracy_metrics']['accuracy'] for r in all_results if r['accuracy_metrics']]
            f1_scores = [r['accuracy_metrics']['f1_score'] for r in all_results if r['accuracy_metrics']]
            precision_scores = [r['accuracy_metrics']['precision'] for r in all_results if r['accuracy_metrics']]
            recall_scores = [r['accuracy_metrics']['recall'] for r in all_results if r['accuracy_metrics']]
            correct_predictions = sum(1 for r in all_results if r['accuracy_metrics'] and r['accuracy_metrics']['is_correct'])
            
            # Token数の集計
            total_tokens = []
            prompt_tokens = []
            completion_tokens = []
            
            for result in all_results:
                for transcript in result.get('interview_transcripts', []):
                    for conversation in transcript.get('conversation_log', []):
                        if 'token_info' in conversation:
                            token_info = conversation['token_info']
                            total_tokens.append(token_info.get('total_tokens', 0))
                            prompt_tokens.append(token_info.get('prompt_tokens', 0))
                            completion_tokens.append(token_info.get('completion_tokens', 0))
            
            # 評価3（知識欠損検出）の集計
            knowledge_gaps_metrics = []
            for r in all_results:
                if r['accuracy_metrics'] and r['accuracy_metrics'].get('knowledge_gaps_metrics'):
                    kg_metrics = r['accuracy_metrics']['knowledge_gaps_metrics']
                    if isinstance(kg_metrics, dict):
                        knowledge_gaps_metrics.append(kg_metrics)
            
            # 評価3の集計指標を計算
            kg_detection_accuracies = [m.get('detection_accuracy', 0) for m in knowledge_gaps_metrics if m.get('detection_accuracy') is not None]
            kg_detection_f1_scores = [m.get('detection_f1_score', 0) for m in knowledge_gaps_metrics if m.get('detection_f1_score') is not None]
            kg_detected_gaps_counts = [m.get('detected_gaps_count', 0) for m in knowledge_gaps_metrics if m.get('detected_gaps_count') is not None]
            kg_true_gaps_counts = [m.get('true_gaps_count', 0) for m in knowledge_gaps_metrics if m.get('true_gaps_count') is not None]
            
            aggregated_metrics = {
                'total_simulations': num_simulations,
                'correct_predictions': correct_predictions,
                'overall_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0,
                'overall_f1_score': np.mean(f1_scores) if f1_scores else 0,
                'overall_precision': np.mean(precision_scores) if precision_scores else 0,
                'overall_recall': np.mean(recall_scores) if recall_scores else 0,
                'accuracy_std': np.std(accuracy_scores) if accuracy_scores else 0,
                'f1_std': np.std(f1_scores) if f1_scores else 0,
                # Token数の集計指標
                'token_usage_metrics': {
                    'total_api_calls': len(total_tokens),
                    'total_tokens_used': sum(total_tokens) if total_tokens else 0,
                    'avg_tokens_per_call': np.mean(total_tokens) if total_tokens else 0,
                    'avg_prompt_tokens': np.mean(prompt_tokens) if prompt_tokens else 0,
                    'avg_completion_tokens': np.mean(completion_tokens) if completion_tokens else 0,
                    'total_prompt_tokens': sum(prompt_tokens) if prompt_tokens else 0,
                    'total_completion_tokens': sum(completion_tokens) if completion_tokens else 0,
                    'tokens_std': np.std(total_tokens) if total_tokens else 0
                },
                # 評価3（知識欠損検出）の集計指標
                'knowledge_gaps_metrics': {
                    'total_simulations_with_kg_data': len(knowledge_gaps_metrics),
                    'avg_detection_accuracy': np.mean(kg_detection_accuracies) if kg_detection_accuracies else 0,
                    'avg_detection_f1_score': np.mean(kg_detection_f1_scores) if kg_detection_f1_scores else 0,
                    'avg_detected_gaps_count': np.mean(kg_detected_gaps_counts) if kg_detected_gaps_counts else 0,
                    'avg_true_gaps_count': np.mean(kg_true_gaps_counts) if kg_true_gaps_counts else 0,
                    'detection_accuracy_std': np.std(kg_detection_accuracies) if kg_detection_accuracies else 0,
                    'detection_f1_std': np.std(kg_detection_f1_scores) if kg_detection_f1_scores else 0
                }
            }
            
            # 全体結果の保存
            final_output = {
                "experiment_summary": {
                    "total_simulations": num_simulations,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "set_index": set_index,  # 元のリクエスト値
                    "interview_flow": all_results[0]['experiment_info']['interview_flow'] if all_results else interview_flow,  # 実際のフローを記録
                    "use_dynamic_flow": use_dynamic_flow or config.USE_INTELLIGENT_DYNAMIC_FLOW,
                    "interviewer_type": interviewer_model_type,
                    "interviewer_model_name": interviewer_model_name
                },
                "aggregated_metrics": aggregated_metrics,
                "individual_results": all_results
            }
            
            filename = f"experiment_summary_{timestamp_str}.json"
            with open(RESULTS_DIR / filename, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            
            experiment_status['results'] = final_output
            update_progress(100, "全シミュレーション完了")
            
            # 最終結果の表示
            log_message(f"\n{'='*80}")
            log_message("=== 全シミュレーション完了 ===")
            log_message(f"総シミュレーション数: {num_simulations}")
            log_message(f"正解予測数: {correct_predictions}/{num_simulations}")
            log_message(f"全体正解率: {aggregated_metrics['overall_accuracy']:.3f} ± {aggregated_metrics['accuracy_std']:.3f}")
            log_message(f"全体F1スコア: {aggregated_metrics['overall_f1_score']:.3f} ± {aggregated_metrics['f1_std']:.3f}")
            
            # Token数統計の表示
            token_metrics = aggregated_metrics['token_usage_metrics']
            log_message(f"\n--- Token数統計 ---")
            log_message(f"総API呼び出し回数: {token_metrics['total_api_calls']}")
            log_message(f"総使用Token数: {token_metrics['total_tokens_used']:,}")
            log_message(f"平均Token数/呼び出し: {token_metrics['avg_tokens_per_call']:.1f} ± {token_metrics['tokens_std']:.1f}")
            log_message(f"総プロンプトToken数: {token_metrics['total_prompt_tokens']:,}")
            log_message(f"総回答Token数: {token_metrics['total_completion_tokens']:,}")
            log_message(f"平均プロンプトToken数: {token_metrics['avg_prompt_tokens']:.1f}")
            log_message(f"平均回答Token数: {token_metrics['avg_completion_tokens']:.1f}")
            
            # 評価3（知識欠損検出）の結果表示
            kg_metrics = aggregated_metrics['knowledge_gaps_metrics']
            log_message(f"\n--- 評価3（知識欠損検出）の集計結果 ---")
            log_message(f"評価3データありシミュレーション数: {kg_metrics['total_simulations_with_kg_data']}/{num_simulations}")
            if kg_metrics['total_simulations_with_kg_data'] > 0:
                log_message(f"平均検出精度: {kg_metrics['avg_detection_accuracy']:.3f} ± {kg_metrics['detection_accuracy_std']:.3f}")
                log_message(f"平均検出F1スコア: {kg_metrics['avg_detection_f1_score']:.3f} ± {kg_metrics['detection_f1_std']:.3f}")
                log_message(f"平均検出知識欠損数: {kg_metrics['avg_detected_gaps_count']:.1f}")
                log_message(f"平均真の知識欠損数: {kg_metrics['avg_true_gaps_count']:.1f}")
            else:
                log_message("評価3のデータがありません")
            
            log_message(f"結果を {filename} に保存しました。")
        
    except Exception as e:
        log_message(f"実験中にエラーが発生しました: {e}")
    finally:
        experiment_status['is_running'] = False

def get_experiment_results():
    """resultsディレクトリから過去の実験結果を取得"""
    results = []
    if RESULTS_DIR.exists():
        # 個別シミュレーション結果
        for file_path in RESULTS_DIR.glob('simulation_*_results_*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        'filename': file_path.name,
                        'type': 'individual',
                        'timestamp': data['experiment_info']['timestamp'],
                        'simulation_num': data['simulation_num'],
                        'dataset_index': data['experiment_info'].get('dataset_index', 'N/A'),
                        'dataset_name': data['experiment_info'].get('dataset_name', 'N/A'),
                        'interviewer_type': data['experiment_info']['interviewer_type'],
                        'interviewer_model_name': data['experiment_info'].get('interviewer_model_name', 'N/A'),
                        'company_name': data['company_profile'].get('name', 'N/A'),
                        'num_candidates': len(data['interview_transcripts']),
                        'accuracy': data['accuracy_metrics']['accuracy'] if data['accuracy_metrics'] else 0,
                        'f1_score': data['accuracy_metrics']['f1_score'] if data['accuracy_metrics'] else 0,
                        'is_correct': data['accuracy_metrics']['is_correct'] if data['accuracy_metrics'] else False,
                        'knowledge_gaps_available': bool(data['accuracy_metrics'].get('knowledge_gaps_metrics')) if data['accuracy_metrics'] else False,
                        'knowledge_gaps_accuracy': data['accuracy_metrics'].get('knowledge_gaps_metrics', {}).get('detection_accuracy', 'N/A') if data['accuracy_metrics'] else 'N/A',
                        'knowledge_gaps_f1': data['accuracy_metrics'].get('knowledge_gaps_metrics', {}).get('detection_f1_score', 'N/A') if data['accuracy_metrics'] else 'N/A'
                    })
            except Exception as e:
                print(f"結果ファイル {file_path} の読み込みに失敗: {e}")
        
        # 実験サマリー結果
        for file_path in RESULTS_DIR.glob('experiment_summary_*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        'filename': file_path.name,
                        'type': 'summary',
                        'timestamp': data['experiment_summary']['timestamp'],
                        'total_simulations': data['experiment_summary']['total_simulations'],
                        'set_index': data['experiment_summary'].get('set_index', 'N/A'),
                        'interviewer_type': data['experiment_summary']['interviewer_type'],
                        'interviewer_model_name': data['experiment_summary'].get('interviewer_model_name', 'N/A'),
                        'overall_accuracy': data['aggregated_metrics']['overall_accuracy'],
                        'overall_f1_score': data['aggregated_metrics']['overall_f1_score'],
                        'correct_predictions': data['aggregated_metrics']['correct_predictions'],
                        'knowledge_gaps_available': bool(data['aggregated_metrics'].get('knowledge_gaps_metrics')),
                        'knowledge_gaps_avg_accuracy': data['aggregated_metrics'].get('knowledge_gaps_metrics', {}).get('avg_detection_accuracy', 'N/A'),
                        'knowledge_gaps_avg_f1': data['aggregated_metrics'].get('knowledge_gaps_metrics', {}).get('avg_detection_f1_score', 'N/A')
                    })
            except Exception as e:
                print(f"結果ファイル {file_path} の読み込みに失敗: {e}")
    
    return sorted(results, key=lambda x: x['timestamp'], reverse=True)

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/api/start_experiment', methods=['POST'])
def start_experiment():
    """実験を開始"""
    if experiment_status['is_running']:
        return jsonify({'error': '実験が既に実行中です'}), 400
    
    data = request.get_json()
    set_index = data.get('set_index') if data.get('set_index') != '' else None
    num_simulations = int(data.get('num_simulations', 1))
    interview_flow = data.get('interview_flow', config.INTERVIEW_FLOW)
    use_dynamic_flow = data.get('use_dynamic_flow', False)
    interviewer_model_type = data.get('interviewer_model_type', config.INTERVIEWER_MODEL_TYPE)
    interviewer_model_name = data.get('interviewer_model_name')
    
    # バックグラウンドで実験を実行
    def run_in_background():
        if interviewer_model_type == 'local':
            local_model, local_tokenizer = initialize_local_model(interviewer_model_name)
            if local_model and local_tokenizer:
                run_experiment_web(
                    local_model, local_tokenizer, set_index, num_simulations, 
                    interview_flow, use_dynamic_flow, interviewer_model_type, interviewer_model_name
                )
            else:
                log_message("ローカルモデルの初期化に失敗したため、実験を中止します。")
        elif interviewer_model_type == 'api':
            run_experiment_web(
                set_index=set_index, num_simulations=num_simulations, 
                interview_flow=interview_flow, use_dynamic_flow=use_dynamic_flow,
                interviewer_model_type=interviewer_model_type, interviewer_model_name=interviewer_model_name
            )
        else:
            log_message(f"エラー: 無効な面接官モデルタイプ '{interviewer_model_type}' が指定されています。")
    
    thread = threading.Thread(target=run_in_background)
    thread.daemon = True
    thread.start()
    
    model_info = f"{interviewer_model_type}モデル ({interviewer_model_name})" if interviewer_model_name else f"{interviewer_model_type}モデル"
    return jsonify({'message': f'{num_simulations}回の実験を開始しました ({model_info}を使用)'})

@app.route('/api/status')
def get_status():
    """実験の状態を取得"""
    return jsonify(experiment_status)

@app.route('/api/results')
def get_results():
    """過去の実験結果一覧を取得"""
    results = get_experiment_results()
    return jsonify(results)

@app.route('/api/results/<filename>')
def get_result_detail(filename):
    """特定の実験結果の詳細を取得"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        return jsonify({'error': 'ファイルが見つかりません'}), 404
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'ファイルの読み込みに失敗: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
