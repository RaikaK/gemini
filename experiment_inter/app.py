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
        if knowledge_gaps_eval and isinstance(knowledge_gaps_eval, dict):
            knowledge_gaps_metrics = knowledge_gaps_eval.get('quantitative_performance_metrics', {})
        
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
            'knowledge_gaps_metrics': knowledge_gaps_metrics
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

def initialize_local_model():
    """Hugging Faceからローカルモデルを読み込み、GPUに配置する"""
    log_message(f"--- 面接官役のローカルモデル ({config.LOCAL_MODEL_NAME}) の初期化を開始 ---")
    
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
        tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(config.LOCAL_MODEL_NAME, quantization_config=quantization_config, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
            
        log_message("--- 面接官役のローカルモデルの初期化完了 ---")
        return model, tokenizer
    except Exception as e:
        log_message(f"モデルの初期化中にエラーが発生しました: {e}")
        return None, None

def run_single_experiment(local_interviewer_model=None, local_interviewer_tokenizer=None, set_index=None, simulation_num=1, interview_flow=None):
    """単一の面接シミュレーション実行"""
    log_message(f"=== シミュレーション {simulation_num} 開始 ===")
    
    # 面接フローが指定されていない場合はデフォルトを使用
    if interview_flow is None:
        interview_flow = config.INTERVIEW_FLOW
    
    # --- 1. db.jsonからデータ読み込み ---
    company_profile, candidate_profiles, actual_set_index = data_generators.load_company_and_candidates_from_db(set_index)
    if company_profile is None or candidate_profiles is None:
        log_message("db.jsonからのデータ読み込みに失敗。実験を中止します。")
        return None

    if len(candidate_profiles) < config.NUM_CANDIDATES:
        log_message(f"警告: 読み込まれた学生数({len(candidate_profiles)})が設定値({config.NUM_CANDIDATES})より少ないため、利用可能な学生を使用します。")
        candidate_profiles = candidate_profiles[:config.NUM_CANDIDATES]

    # --- 2. 面接官と候補者の初期化 ---
    model_type = config.INTERVIEWER_MODEL_TYPE
    log_message(f"--- 面接官タイプ: {model_type} ---")
    
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
            log_message(f"エラー: config.pyのINTERVIEWER_MODEL_TYPEに無効な値 '{model_type}' が設定されています。")
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
                answer = applicant.generate(
                    state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                )
                log_message(f"学生 (API): {answer}")
                state["conversation_log"].append({"turn": round_num + 1, "question": question, "answer": answer})

        elif question_type == 1: # 個別質問
            log_message("--- 個別質問フェーズ ---")
            for i, state in enumerate(candidate_states):
                log_message(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                question, _ = interviewer.ask_question(state["conversation_log"])
                log_message(f"面接官 ({model_type}): {question}")
                answer = applicant.generate(
                    state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                )
                log_message(f"学生 (API): {answer}")
                state["conversation_log"].append({"turn": round_num + 1, "question": question, "answer": answer})

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
            "interviewer_type": model_type,
            "interview_flow": interview_flow,
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
            log_message(f"評価3 - 知識欠損検出: 利用可能")
    
    return result

def run_experiment_web(local_interviewer_model=None, local_interviewer_tokenizer=None, set_index=None, num_simulations=1, interview_flow=None):
    """Web用の面接シミュレーション実行関数（複数回対応）"""
    try:
        experiment_status['is_running'] = True
        experiment_status['logs'] = []
        experiment_status['results'] = None
        experiment_status['simulation_results'] = []
        experiment_status['current_simulation'] = 0
        experiment_status['total_simulations'] = num_simulations
        
        update_progress(0, f"{num_simulations}回のシミュレーションを開始しています...")
        
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
            result = run_single_experiment(local_interviewer_model, local_interviewer_tokenizer, set_index, sim_num, interview_flow)
            
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
            
            aggregated_metrics = {
                'total_simulations': num_simulations,
                'correct_predictions': correct_predictions,
                'overall_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0,
                'overall_f1_score': np.mean(f1_scores) if f1_scores else 0,
                'overall_precision': np.mean(precision_scores) if precision_scores else 0,
                'overall_recall': np.mean(recall_scores) if recall_scores else 0,
                'accuracy_std': np.std(accuracy_scores) if accuracy_scores else 0,
                'f1_std': np.std(f1_scores) if f1_scores else 0
            }
            
            # 全体結果の保存
            final_output = {
                "experiment_summary": {
                    "total_simulations": num_simulations,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "set_index": set_index,  # 元のリクエスト値
                    "interview_flow": interview_flow,
                    "interviewer_type": config.INTERVIEWER_MODEL_TYPE
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
                        'company_name': data['company_profile'].get('name', 'N/A'),
                        'num_candidates': len(data['interview_transcripts']),
                        'accuracy': data['accuracy_metrics']['accuracy'] if data['accuracy_metrics'] else 0,
                        'f1_score': data['accuracy_metrics']['f1_score'] if data['accuracy_metrics'] else 0,
                        'is_correct': data['accuracy_metrics']['is_correct'] if data['accuracy_metrics'] else False
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
                        'overall_accuracy': data['aggregated_metrics']['overall_accuracy'],
                        'overall_f1_score': data['aggregated_metrics']['overall_f1_score'],
                        'correct_predictions': data['aggregated_metrics']['correct_predictions']
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
    
    # バックグラウンドで実験を実行
    def run_in_background():
        model_type = config.INTERVIEWER_MODEL_TYPE
        
        if model_type == 'local':
            local_model, local_tokenizer = initialize_local_model()
            if local_model and local_tokenizer:
                run_experiment_web(local_model, local_tokenizer, set_index, num_simulations, interview_flow)
            else:
                log_message("ローカルモデルの初期化に失敗したため、実験を中止します。")
        elif model_type == 'api':
            run_experiment_web(set_index=set_index, num_simulations=num_simulations, interview_flow=interview_flow)
        else:
            log_message(f"エラー: config.pyのINTERVIEWER_MODEL_TYPEに無効な値 '{model_type}' が設定されています。")
    
    thread = threading.Thread(target=run_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'{num_simulations}回の実験を開始しました'})

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
