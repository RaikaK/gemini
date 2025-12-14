# runner.py - 実行ロジック本体

import json
import datetime
import random
import time
from pathlib import Path
import re

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandbモジュールが見つかりません。計測の記録をスキップします。")

from config import (
    INTERVIEWER_MODEL, APPLICANT_MODEL, NUM_CANDIDATES, MAX_ROUNDS,
    ASPIRATION_LEVEL_MAPPING, DB_FILE_PATH,
    INTERVIEWER_MODEL_TYPE, LOCAL_MODEL_NAME
)
from interviewer import Interviewer
from student import CompanyKnowledgeManager, Applicant
from metrics import (
    calculate_ranking_accuracy,
    calculate_knowledge_gaps_metrics,
    get_last_common_question_evaluations,
    get_last_common_question_ranking_accuracy,
    get_last_common_question_knowledge_gaps_metrics,
)

# ローカルモデル管理のインポート（オプション）
try:
    from model_manager import HuggingFaceModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    print("警告: model_managerモジュールが見つかりません。ローカルモデルは使用できません。")


def generate_experiment_id(set_index, interviewer_model_type, interviewer_model_name, max_rounds=None):
    """実験IDを生成する
    
    Args:
        set_index: データセットのインデックス
        interviewer_model_type: 面接官モデルのタイプ
        interviewer_model_name: 面接官モデル名
        max_rounds: 最大ラウンド数
    
    Returns:
        実験ID文字列
    """
    import hashlib
    
    # 実験設定を文字列に変換
    config_str = f"{set_index}_{interviewer_model_type}_{interviewer_model_name}_{max_rounds}"
    
    # ハッシュ化して短いIDを生成
    hash_obj = hashlib.md5(config_str.encode())
    hash_hex = hash_obj.hexdigest()[:8]
    
    # 読みやすい形式に変換
    model_short = interviewer_model_name.split('/')[-1] if '/' in interviewer_model_name else interviewer_model_name
    model_short = model_short.replace('-', '_').replace('.', '_')[:20]
    
    set_str = f"set{set_index}" if set_index is not None else "random"
    rounds_str = f"r{max_rounds}" if max_rounds else "r20"
    
    experiment_id = f"{set_str}_{interviewer_model_type}_{model_short}_{rounds_str}_{hash_hex}"
    return experiment_id


def find_existing_summary_file(results_dir, experiment_id):
    """既存のサマリーファイルを検索する
    
    Args:
        results_dir: 結果ディレクトリのパス
        experiment_id: 実験ID
    
    Returns:
        既存のサマリーファイルのパス、見つからない場合はNone
    """
    # experiment_idを含むファイルを検索
    pattern = f"experiment_summary_*_{experiment_id}.json"
    existing_files = list(results_dir.glob(pattern))
    
    if existing_files:
        # 最新のファイルを返す
        return max(existing_files, key=lambda p: p.stat().st_mtime)
    return None


def _format_gap_summary(missing_predictions):
    """欠損予測のサマリ文字列を生成（評価1/2用の参考情報）"""
    lines = []
    for cname, mp in missing_predictions.items():
        missing = [k for k, v in mp.items() if v]
        known = [k for k, v in mp.items() if not v]
        lines.append(f"{cname}: missing={missing if missing else 'none'}, known={known if known else 'none'}")
    return "\n".join(lines) if lines else "（まだ欠損推定はありません）"


def _safe_wandb_log(wandb_run, data, step=None):
    """wandb.logのラッパー（wandb未導入や失敗時は黙ってスキップ）"""
    if wandb_run is None or not data:
        return
    try:
        wandb_run.log(data, step=step)
    except Exception as e:
        print(f"警告: wandb.log中にエラーが発生しました: {e}")


def _avg(values):
    """Noneを除外して平均を計算（値がなければ0.0）"""
    cleaned = [v for v in values if v is not None]
    return sum(cleaned) / len(cleaned) if cleaned else 0.0


def _compute_knowledge_gap_metrics(question_keys, candidate_states, missing_predictions):
    """欠損予測の進捗から精度指標を計算する（未予測キーは欠損なし扱い）"""
    tp = tn = fp = fn = 0
    for state in candidate_states:
        cname = state['profile']['name']
        for k in question_keys:
            actual_missing = not bool(state['knowledge'].get(k, ""))
            predicted_missing = missing_predictions.get(cname, {}).get(k, False)
            tp += int(predicted_missing and actual_missing)
            tn += int((not predicted_missing) and (not actual_missing))
            fp += int(predicted_missing and (not actual_missing))
            fn += int((not predicted_missing) and actual_missing)
    denom = tp + tn + fp + fn
    accuracy = (tp + tn) / denom if denom else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_data_from_db(set_index=None):
    """db.jsonからデータを読み込む"""
    try:
        # configから相対パスを取得
        db_path = Path(__file__).parent / DB_FILE_PATH
        
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print("エラー: db.jsonの形式が不正です")
            return None, None, None
        
        # セットインデックスが指定されていない場合はランダムに選択（21-39を除外）
        allowed_indices = [i for i in range(len(data)) if not (21 <= i <= 39)]
        if set_index is None:
            set_index = random.choice(allowed_indices)
        elif set_index >= len(data) or not (0 <= set_index < len(data)):
            print(f"警告: 指定されたインデックス {set_index} が範囲外です。許可された範囲から再選択します。")
            set_index = random.choice(allowed_indices)
        elif 21 <= set_index <= 39:
            print(f"警告: インデックス {set_index} は使用禁止範囲(21-39)です。許可された範囲から再選択します。")
            set_index = random.choice(allowed_indices)
        
        selected_set = data[set_index]
        
        company_profile = selected_set['company']
        candidate_profiles = selected_set['students']
        
        # 学生のpreparationレベルを設定
        for profile in candidate_profiles:
            aspiration_level = profile.get('aspiration_level', 'medium_70_percent')
            # マッピングから取得、見つからない場合はlowをデフォルトに
            profile['preparation'] = ASPIRATION_LEVEL_MAPPING.get(
                aspiration_level, 
                'low'  # デフォルト値
            )
        
        random.shuffle(candidate_profiles)
        
        print(f"\n=== セット {set_index + 1} を選択 ===")
        print(f"企業: {company_profile.get('name', 'N/A')}")
        print(f"学生数: {len(candidate_profiles)}人")
        
        return company_profile, candidate_profiles[:NUM_CANDIDATES], set_index
        
    except FileNotFoundError:
        print("エラー: db.jsonが見つかりません")
        return None, None, None
    except Exception as e:
        print(f"エラー: データ読み込み失敗 - {e}")
        return None, None, None





def initialize_local_model(model_key=None):
    """ローカルモデルを初期化"""
    if not MODEL_MANAGER_AVAILABLE:
        print("エラー: ローカルモデル管理モジュールが利用できません")
        return None, None
    
    if model_key is None:
        model_key = LOCAL_MODEL_NAME
    
    print(f"--- ローカルモデル ({model_key}) の初期化を開始 ---")
    
    model_manager = HuggingFaceModelManager()
    
    # huggingface_hub モジュールの確認
    if not model_manager.check_hf_cli_installed():
        print("huggingface_hubがインストールされていません。インストール中...")
        if not model_manager.install_hf_cli():
            print("huggingface_hubのインストールに失敗しました")
            return None, None

    # モデルのダウンロード確認
    if not model_manager.is_model_downloaded(model_key):
        model_info = model_manager.get_model_info(model_key)
        if model_info:
            print(f"モデル {model_key} が初回選択されました。ダウンロードを開始します...")
            print(f"モデルサイズ: {model_info['size_gb']}GB")
            print(f"推奨GPU: {model_info['recommended_gpu']}")
            print("ダウンロードには時間がかかる場合があります。しばらくお待ちください...")
            
            def progress_callback(message):
                print(f"[ダウンロード進捗] {message}")
            
            if not model_manager.download_model(model_key, progress_callback=progress_callback):
                print(f"モデル {model_key} のダウンロードに失敗しました")
                return None, None
            else:
                print(f"モデル {model_key} のダウンロードが完了しました")
        else:
            print(f"モデル {model_key} の情報を取得できませんでした")
            return None, None
    else:
        print(f"モデル {model_key} は既にダウンロード済みです")
    
    # モデルの初期化
    print(f"モデル {model_key} を初期化中...")
    model, tokenizer = model_manager.initialize_model(model_key)
    
    if model and tokenizer:
        print("--- ローカルモデルの初期化完了 ---")
    else:
        print("--- ローカルモデルの初期化に失敗 ---")
    
    return model, tokenizer


def _accumulate_tokens(token_totals, token_info):
    """ラウンド内でのトークン数を集計するヘルパー"""
    if not token_info:
        return
    token_totals['prompt_tokens'] += token_info.get('prompt_tokens', 0)
    token_totals['completion_tokens'] += token_info.get('completion_tokens', 0)
    token_totals['total_tokens'] += token_info.get('total_tokens', 0)


def _extract_candidate_from_text(candidate_states, evaluation_text):
    """評価テキストから候補者名を推定するヘルパー"""
    if not evaluation_text:
        return None
    for state in candidate_states:
        candidate_name = state['profile']['name']
        if candidate_name in evaluation_text:
            return state
    match = re.search(r'(学生[A-Z]{1,3}\d{0,2})', evaluation_text)
    if match:
        extracted_name = match.group(1)
        for state in candidate_states:
            if state['profile']['name'] == extracted_name:
                return state
    return None


def _get_true_lowest_candidates(candidate_states):
    """準備レベルから真の「志望度が低い」候補者集合を返す"""
    level_score = {'low': 1, 'medium': 2, 'high': 3}
    scores = []
    for state in candidate_states:
        prep = state['profile'].get('preparation', 'low')
        scores.append((state['profile']['name'], level_score.get(prep, 1)))
    min_score = min(score for _, score in scores)
    return {name for name, score in scores if score == min_score}


def _is_least_motivated_prediction_correct(candidate_states, least_motivated_eval):
    """評価1の予測が正しいかを1/0で返す"""
    true_lowest = _get_true_lowest_candidates(candidate_states)
    pred_state = _extract_candidate_from_text(candidate_states, least_motivated_eval)
    if pred_state is None:
        return 0
    return 1 if pred_state['profile']['name'] in true_lowest else 0

def run_single_interview(set_index=None, simulation_num=1, interviewer_model_type=None, interviewer_model_name=None, max_rounds=None, local_model=None, local_tokenizer=None, api_provider=None, experiment_id=None, wandb_group=None):
    """単一の面接シミュレーションを実行
    
    Args:
        set_index: データセットのインデックス
        simulation_num: シミュレーション番号
        interviewer_model_type: 面接官モデルのタイプ ('local' または 'api')
        interviewer_model_name: 面接官モデル名
        max_rounds: 最大ラウンド数
        local_model: 再利用するローカルモデル（オプション、提供されない場合は新規初期化）
        local_tokenizer: 再利用するローカルトークナイザー（オプション、提供されない場合は新規初期化）
        experiment_id: 実験ID（wandbログ用、任意）
        wandb_group: wandbのグループ名（同じ実験IDなどを指定）
    """
    print("\n" + "="*60)
    print(f"面接ロールプレイ実行システム - シミュレーション {simulation_num}")
    print("="*60)
    
    # データ読み込み
    company_profile, candidate_profiles, actual_set_index = load_data_from_db(set_index)
    
    if company_profile is None or candidate_profiles is None:
        print("データ読み込みに失敗しました")
        return None

    # ラウンド数の設定（企業キー数に合わせる）
    question_keys = [k for k in company_profile.keys() if k not in ("id", "name")]
    max_rounds = len(question_keys)
    
    # モデル設定の決定
    if interviewer_model_type is None:
        interviewer_model_type = INTERVIEWER_MODEL_TYPE
    
    if interviewer_model_name is None:
        if interviewer_model_type == 'local':
            interviewer_model_name = LOCAL_MODEL_NAME
        else:
            interviewer_model_name = INTERVIEWER_MODEL
    
    # 面接官を初期化
    if interviewer_model_type == 'local':
        # 既に初期化されたモデルが提供されている場合は再利用、そうでない場合は新規初期化
        if local_model is None or local_tokenizer is None:
            local_model, local_tokenizer = initialize_local_model(interviewer_model_name)
            if local_model is None or local_tokenizer is None:
                print("ローカルモデルの初期化に失敗しました。")
                return None
        else:
            print(f"--- ローカルモデル ({interviewer_model_name}) を再利用 ---")
        
        interviewer = Interviewer(
            company_profile,
            model_type='local',
            model=local_model,
            tokenizer=local_tokenizer,
            local_model_key=interviewer_model_name,
            api_provider=api_provider
        )
        print(f"--- 面接官タイプ: ローカルモデル ({interviewer_model_name}) ---")
    else:
        interviewer = Interviewer(
            company_profile,
            model_name=interviewer_model_name,
            model_type='api',
            api_provider=api_provider
        )
        print(f"--- 面接官タイプ: APIモデル ({interviewer_model_name}) ---")
    
    # 応募者を初期化
    applicant = Applicant(APPLICANT_MODEL)
    knowledge_manager = CompanyKnowledgeManager(company_profile, experiment_id=experiment_id)
    
    # 質問に使う企業キー（固定リスト）
    question_keys = knowledge_manager.question_keys
    # 候補者の状態を初期化
    candidate_states = []
    for profile in candidate_profiles:
        knowledge_dict, coverage_str = knowledge_manager.get_knowledge_for_level(
            profile.get('preparation', 'low'),
            candidate_name=profile.get('name')
        )
        candidate_states.append({
            'profile': profile,
            'knowledge': knowledge_dict,
            'knowledge_coverage': coverage_str,
            'conversation_log': [],
            'remaining_keys': question_keys.copy()
        })
        print(f"\n候補者: {profile['name']} (準備レベル: {profile['preparation']}, 知識カバレッジ: {coverage_str})")
    
    # 面接ラウンドを実行
    asked_common_questions = []  # 全体質問の履歴
    asked_individual_questions = []  # 個別質問の履歴（候補者ごと）
    remaining_common_keys = question_keys.copy()
    
    print(f"\n{'='*60}")
    print(f"面接開始（{max_rounds}ラウンド: 全体質問と個別質問を交互に実施）")
    print(f"{'='*60}\n")
    
    # ラウンドごとの評価結果を保存
    round_evaluations = []
    
    # 直前の全体質問で最も志望度が低いと判断された候補者（個別質問の対象）
    target_candidate_for_individual = None

    # ラウンドごとの配列（wandb送信用）
    eval1_hits_series = []
    eval2_hits_series = []
    per_round_metrics = []
    # 欠損予測の累積 (candidate -> {key -> missing_bool})
    missing_predictions = {state['profile']['name']: {} for state in candidate_states}

    # シミュレーション全体のトークン集計
    simulation_token_totals = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }

    # wandb Run（ラウンド単位のログを送るため先に初期化）
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            run_name = f"{interviewer_model_name}_sim{simulation_num}_set{actual_set_index}"
            group_name = wandb_group or f"interviewer_model_name:{interviewer_model_name}"
            wandb_run = wandb.init(
                project="mochi-interview",
                name=run_name,
                config={
                    "simulation_num": simulation_num,
                    "set_index": actual_set_index,
                    "interviewer_model_type": interviewer_model_type,
                    "interviewer_model_name": interviewer_model_name,
                    "applicant_model_name": APPLICANT_MODEL,
                    "max_rounds": max_rounds,
                    "api_provider": api_provider,
                    "experiment_id": experiment_id,
                },
                group=group_name,
                tags=[
                    f"interviewer_model_name:{interviewer_model_name}",
                    f"applicant_model_name:{APPLICANT_MODEL}",
                    f"set:{actual_set_index}",
                ],
                reinit=True,
            )
        except Exception as e:
            wandb_run = None
            print(f"警告: wandbの初期化に失敗しました: {e}")
    
    for round_num in range(1, max_rounds + 1):
        round_start_time = time.time()
        round_token_info = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        current_round_metrics = {
            'round': round_num,
            'eval1/accuracy': 0.0,
            'eval1/f1': 0.0,
            'eval2/hits': 0.0,
            'eval3/accuracy': 0.0,
            'eval3/f1': 0.0,
            'eval3/precision': 0.0,
            'eval3/recall': 0.0,
        }
        
        # 全ラウンドを全体質問に統一
        is_common_question = True
        
        if is_common_question:
            # 全体質問
            print(f"\n--- ラウンド {round_num} (全体質問) ---")
        
            # 全体質問を生成（まだ聞いていない企業項目から選択）
            target_key_for_common = remaining_common_keys.pop(0) if remaining_common_keys else question_keys[0]
            question_start_time = time.time()
            question, question_token_info = interviewer.ask_common_question(target_key_for_common)
            question_time = time.time() - question_start_time
            
            _accumulate_tokens(round_token_info, question_token_info)
            
            asked_common_questions.append(question)
            print(f"\n[質問項目] {target_key_for_common}")
            print(f"面接官の質問（全員へ）: {question}")
            
        # 各候補者が回答
        for state in candidate_states:
            answer_start_time = time.time()
            answer, token_info = applicant.generate_answer(
                state['profile'],
                state['knowledge'],
                state['conversation_log'],
                question
            )
            answer_time = time.time() - answer_start_time
            
            _accumulate_tokens(round_token_info, token_info)
            
            # 会話ログに記録
            state['conversation_log'].append({
                'round': round_num,
                'question_type': 'common' if is_common_question else 'individual',
                'question': question,
                'answer': answer,
                'token_info': token_info
            })
            
            print(f"\n{state['profile']['name']}: {answer}")
    
        # 全体質問後の評価（評価1、2、3を実行）
        if is_common_question:
            print(f"\n{'='*60}")
            print(f"ラウンド {round_num} (全体質問) 終了後の評価")
            print(f"{'='*60}\n")
            
            # 評価1: 最も志望度が低い候補者を選定
            print(f"【ラウンド {round_num} - 評価1: 最も志望度が低い候補者の選定】")
            eval1_start_time = time.time()
            least_motivated_eval, eval1_token_info = interviewer.select_least_motivated_candidate(candidate_states)
            eval1_time = time.time() - eval1_start_time
            _accumulate_tokens(round_token_info, eval1_token_info)
            print(f"{least_motivated_eval}\n")
            eval1_hit = _is_least_motivated_prediction_correct(candidate_states, least_motivated_eval)
            eval1_hits_series.append(eval1_hit)
            current_round_metrics['eval1/accuracy'] = eval1_hit
            current_round_metrics['eval1/f1'] = eval1_hit
            target_state = _extract_candidate_from_text(candidate_states, least_motivated_eval)
            if target_state:
                target_candidate_for_individual = target_state
                print(f"次の個別質問の対象: {target_state['profile']['name']}")

            print(f"{'='*60}\n")

            # 評価2: ランキング
            print(f"【ラウンド {round_num} - 評価2: 志望度ランキング（低い順）】")
            eval2_start_time = time.time()
            ranking_eval, eval2_token_info = interviewer.rank_candidates_by_motivation(candidate_states)
            eval2_time = time.time() - eval2_start_time
            _accumulate_tokens(round_token_info, eval2_token_info)
            print(f"{ranking_eval}\n")
            ranking_accuracy = calculate_ranking_accuracy(candidate_states, ranking_eval)
            perfect_match = 1 if ranking_accuracy and ranking_accuracy.get('is_valid') and ranking_accuracy.get('correct_positions') == ranking_accuracy.get('total_positions') else 0
            eval2_hits_series.append(perfect_match)
            if ranking_accuracy:
                if ranking_accuracy.get('is_valid'):
                    print(f"精度スコア: {ranking_accuracy['accuracy']:.3f}")
                else:
                    print(f"警告: {ranking_accuracy.get('message', 'ランキングが正しく抽出できませんでした')}")
            current_round_metrics['eval2/hits'] = perfect_match

            # 評価3: 知識欠損検出
            print(f"【ラウンド {round_num} - 評価3: 知識欠損検出】")
            eval3_start_time = time.time()
            knowledge_gaps_eval, eval3_token_info = interviewer.detect_knowledge_gaps(
                candidate_states,
                None,
                None,
                target_keys=[target_key_for_common]
            )
            eval3_time = time.time() - eval3_start_time
            if eval3_token_info:
                round_token_info['prompt_tokens'] += eval3_token_info.get('prompt_tokens', 0)
                round_token_info['completion_tokens'] += eval3_token_info.get('completion_tokens', 0)
                round_token_info['total_tokens'] += eval3_token_info.get('total_tokens', 0)
            per_pred = knowledge_gaps_eval.get("per_candidate_predictions", {}) if knowledge_gaps_eval else {}
            for state in candidate_states:
                cname = state['profile']['name']
                pred = per_pred.get(cname, {})
                missing_predictions[cname][target_key_for_common] = bool(pred.get("missing", False))
            # ラウンド時点の欠損判定指標（未予測キーは欠損なしで計算）
            kg_metrics_progress = _compute_knowledge_gap_metrics(question_keys, candidate_states, missing_predictions)
            current_round_metrics.update({
                'eval3/accuracy': kg_metrics_progress['accuracy'],
                'eval3/f1': kg_metrics_progress['f1_score'],
                'eval3/precision': kg_metrics_progress['precision'],
                'eval3/recall': kg_metrics_progress['recall'],
            })

            # ラウンドの評価結果を保存
            round_evaluations.append({
                'round': round_num,
                'question_type': 'common',
                'evaluations': {
                    'least_motivated': least_motivated_eval,
                    'ranking': ranking_eval,
                    'knowledge_gaps': knowledge_gaps_eval
                },
                'eval1_hit': eval1_hit,
                'eval2_perfect_match': perfect_match,
                'ranking_accuracy': ranking_accuracy,
                'knowledge_gaps_predictions': per_pred
            })
        else:
            # 個別質問
            print(f"\n--- ラウンド {round_num} (個別質問) ---")
            
            # 対象候補者が決定されているか確認
            if target_candidate_for_individual is None:
                print("警告: 個別質問の対象候補者が決定されていません。最初の候補者に質問します。")
                target_candidate_for_individual = candidate_states[0]
            
            target_candidate_name = target_candidate_for_individual['profile']['name']
            print(f"対象候補者: {target_candidate_name}")
            # この候補者にまだ聞いていない項目から選ぶ（尽きたら全項目にリセット）
            remaining_keys = target_candidate_for_individual.get('remaining_keys', [])
            if not remaining_keys:
                remaining_keys = question_keys.copy()
                target_candidate_for_individual['remaining_keys'] = remaining_keys
            target_key_for_individual = remaining_keys.pop(0)
            
            # 個別質問を生成（対象候補者の会話履歴を使用）
            question_start_time = time.time()
            question, question_token_info = interviewer.ask_question(
                target_key_for_individual,
                target_candidate_for_individual['conversation_log']
            )
            question_time = time.time() - question_start_time
            
            _accumulate_tokens(round_token_info, question_token_info)
            
            asked_individual_questions.append({
                'candidate': target_candidate_name,
                'question': question
            })
            print(f"\n面接官の質問（{target_candidate_name}へ）: {question}")
            
            # 対象候補者のみ回答
            answer_start_time = time.time()
            answer, token_info = applicant.generate_answer(
                target_candidate_for_individual['profile'],
                target_candidate_for_individual['knowledge'],
                target_candidate_for_individual['conversation_log'],
                question
            )
            answer_time = time.time() - answer_start_time
            
            _accumulate_tokens(round_token_info, token_info)
            
            # 会話ログに記録
            target_candidate_for_individual['conversation_log'].append({
                'round': round_num,
                'question_type': 'individual',
                'question': question,
                'answer': answer,
                'token_info': token_info
            })
            
            print(f"\n{target_candidate_name}: {answer}")
            # 個別質問後の評価（すべての個別質問ラウンドで評価を実行）
            print(f"\n{'='*60}")
            print(f"ラウンド {round_num} (個別質問) 終了後の評価")
            print(f"{'='*60}\n")
            
            # 評価1: 最も志望度が低い候補者を選定
            print(f"【ラウンド {round_num} - 評価1: 最も志望度が低い候補者の選定】")
            eval1_start_time = time.time()
            least_motivated_eval, eval1_token_info = interviewer.select_least_motivated_candidate(candidate_states)
            eval1_time = time.time() - eval1_start_time
            
            # token数を集計
            if eval1_token_info:
                round_token_info['prompt_tokens'] += eval1_token_info.get('prompt_tokens', 0)
                round_token_info['completion_tokens'] += eval1_token_info.get('completion_tokens', 0)
                round_token_info['total_tokens'] += eval1_token_info.get('total_tokens', 0)
            
            print(f"{least_motivated_eval}\n")

            # 評価1正解判定（1/0）
            eval1_hit = _is_least_motivated_prediction_correct(candidate_states, least_motivated_eval)
            eval1_hits_series.append(eval1_hit)
            current_round_metrics['eval1/accuracy'] = eval1_hit
            current_round_metrics['eval1/f1'] = eval1_hit
            
            # 評価1の結果から候補者名を抽出（次の全体質問での個別質問の対象として保存）
            import re
            # より柔軟なパターンで候補者名を抽出
            next_target_candidate_name = None
            # パターン1: 直接候補者名が含まれている場合
            for state in candidate_states:
                candidate_name = state['profile']['name']
                # 候補者名がテキストに含まれているかチェック
                if candidate_name in least_motivated_eval:
                    next_target_candidate_name = candidate_name
                    target_candidate_for_individual = state
                    print(f"次の個別質問の対象: {next_target_candidate_name}")
                    break
            
            # パターン2: 正規表現で抽出を試みる
            if next_target_candidate_name is None:
                # 「学生」で始まり、英数字が続くパターン
                match = re.search(r'(学生[A-Z]{1,3}\d{0,2})', least_motivated_eval)
                if match:
                    extracted_name = match.group(1)
                    # 候補者名リストと照合
                    for state in candidate_states:
                        if state['profile']['name'] == extracted_name:
                            next_target_candidate_name = extracted_name
                            target_candidate_for_individual = state
                            print(f"次の個別質問の対象: {next_target_candidate_name} (正規表現で抽出)")
                            break
            
            # 評価1と評価2の区切り
            print(f"{'='*60}\n")
            
            # 評価2: ランキング
            print(f"【ラウンド {round_num} - 評価2: 志望度ランキング（低い順）】")
            eval2_start_time = time.time()
            ranking_eval, eval2_token_info = interviewer.rank_candidates_by_motivation(candidate_states)
            eval2_time = time.time() - eval2_start_time
            
            # token数を集計
            if eval2_token_info:
                round_token_info['prompt_tokens'] += eval2_token_info.get('prompt_tokens', 0)
                round_token_info['completion_tokens'] += eval2_token_info.get('completion_tokens', 0)
                round_token_info['total_tokens'] += eval2_token_info.get('total_tokens', 0)
            
            print(f"{ranking_eval}\n")

            # 評価2: ランキング精度の計算
            ranking_accuracy = calculate_ranking_accuracy(candidate_states, ranking_eval)
            perfect_match = 1 if ranking_accuracy and ranking_accuracy.get('is_valid') and ranking_accuracy.get('correct_positions') == ranking_accuracy.get('total_positions') else 0
            eval2_hits_series.append(perfect_match)
            if ranking_accuracy:
                if ranking_accuracy.get('is_valid'):
                    print(f"精度スコア: {ranking_accuracy['accuracy']:.3f}")
                else:
                    print(f"警告: {ranking_accuracy.get('message', 'ランキングが正しく抽出できませんでした')}")
            current_round_metrics['eval2/hits'] = perfect_match
            
            # 評価2と評価3の区切り
            print(f"{'='*60}\n")
            
            # 評価3: 知識欠損検出（予測のみ保存）
            print(f"【ラウンド {round_num} - 評価3: 知識欠損検出】")
            eval3_start_time = time.time()
            knowledge_gaps_eval, eval3_token_info = interviewer.detect_knowledge_gaps(
                candidate_states,
                least_motivated_eval,
                ranking_eval,
                target_keys=[target_key_for_common]
            )
            eval3_time = time.time() - eval3_start_time
            
            # token数を集計
            if eval3_token_info:
                round_token_info['prompt_tokens'] += eval3_token_info.get('prompt_tokens', 0)
                round_token_info['completion_tokens'] += eval3_token_info.get('completion_tokens', 0)
                round_token_info['total_tokens'] += eval3_token_info.get('total_tokens', 0)

            # 欠損予測のみ保存（集計は全ラウンド後に一括で実施）
            per_pred = knowledge_gaps_eval.get("per_candidate_predictions", {}) if knowledge_gaps_eval else {}
            for state in candidate_states:
                cname = state['profile']['name']
                pred = per_pred.get(cname, {})
                missing_predictions[cname][target_key_for_common] = bool(pred.get("missing", False))
            kg_metrics_progress = _compute_knowledge_gap_metrics(question_keys, candidate_states, missing_predictions)
            current_round_metrics.update({
                'eval3/accuracy': kg_metrics_progress['accuracy'],
                'eval3/f1': kg_metrics_progress['f1_score'],
                'eval3/precision': kg_metrics_progress['precision'],
                'eval3/recall': kg_metrics_progress['recall'],
            })
            
            # ラウンドの評価結果を保存（共通質問のみ）
            round_evaluations.append({
                'round': round_num,
                'question_type': 'common',
                'evaluations': {
                    'least_motivated': least_motivated_eval,
                    'ranking': ranking_eval,
                    'knowledge_gaps': knowledge_gaps_eval
                },
                'eval1_hit': eval1_hit,
                'eval2_perfect_match': perfect_match,
                'ranking_accuracy': ranking_accuracy,
                'knowledge_gaps_predictions': per_pred
            })

        # ラウンドメトリクスを保存（プロットやデバッグ用）
        per_round_metrics.append(current_round_metrics)

        # ラウンドごとのwandbログ（eval1/2のみ）
        if wandb_run:
            _safe_wandb_log(
                wandb_run,
                {
                    'eval1/hits': eval1_hits_series[-1] if eval1_hits_series else None,
                    'eval2/hits': eval2_hits_series[-1] if eval2_hits_series else None,
                    'eval2/accuracy': ranking_accuracy.get('accuracy') if (ranking_accuracy and ranking_accuracy.get('is_valid')) else None,
                    'eval3/accuracy': current_round_metrics.get('eval3/accuracy'),
                    'eval3/f1': current_round_metrics.get('eval3/f1'),
                    'eval3/precision': current_round_metrics.get('eval3/precision'),
                    'eval3/recall': current_round_metrics.get('eval3/recall'),
                },
                step=round_num,
            )

        # ラウンド間の区切り
        if round_num < max_rounds:
            print(f"\n{'='*60}")
            print(f"ラウンド {round_num} 完了。次のラウンドに進みます...")
            print(f"{'='*60}\n")

        # ラウンドごとのトークンをシミュレーショントータルに集計
        simulation_token_totals['prompt_tokens'] += round_token_info.get('prompt_tokens', 0)
        simulation_token_totals['completion_tokens'] += round_token_info.get('completion_tokens', 0)
        simulation_token_totals['total_tokens'] += round_token_info.get('total_tokens', 0)
    
    # 結果を保存
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 評価3の集計（全項目終了後にまとめて計算）
    kg_overall = {}
    tp_total = tn_total = fp_total = fn_total = 0
    for state in candidate_states:
        cname = state['profile']['name']
        kg_overall[cname] = {}
        for k in question_keys:
            actual_missing = not bool(state['knowledge'].get(k, ""))
            predicted_missing = missing_predictions.get(cname, {}).get(k, False)
            tp = int(predicted_missing and actual_missing)
            tn = int((not predicted_missing) and (not actual_missing))
            fp = int(predicted_missing and (not actual_missing))
            fn = int((not predicted_missing) and actual_missing)
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn
            kg_overall[cname][k] = {
                "actual_missing": actual_missing,
                "predicted_missing": predicted_missing,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }
    denom_total = tp_total + tn_total + fp_total + fn_total
    acc_total = (tp_total + tn_total) / denom_total if denom_total else 0.0
    prec_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    rec_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1_total = 2 * prec_total * rec_total / (prec_total + rec_total) if (prec_total + rec_total) else 0.0
    knowledge_gap_metrics_overall = {
        "accuracy": round(acc_total, 3),
        "precision": round(prec_total, 3),
        "recall": round(rec_total, 3),
        "f1_score": round(f1_total, 3),
        "tp": tp_total, "tn": tn_total, "fp": fp_total, "fn": fn_total,
    }
    result_data = {
        'simulation_num': simulation_num,
        'timestamp': datetime.datetime.now().isoformat(),
        'set_index': actual_set_index,
        'interviewer_model_type': interviewer_model_type,
        'interviewer_model_name': interviewer_model_name,
        'company_profile': company_profile,
        'interview_transcripts': [
            {
                'candidate': state['profile']['name'],
                'preparation': state['profile']['preparation'],
                'knowledge_coverage': state['knowledge_coverage'],
                'conversation_log': state['conversation_log']
            }
            for state in candidate_states
        ],
        'evaluations': get_last_common_question_evaluations(round_evaluations),
        'ranking_accuracy': get_last_common_question_ranking_accuracy(round_evaluations),
        'knowledge_gaps_metrics': get_last_common_question_knowledge_gaps_metrics(round_evaluations),
        'eval1_hits': {re['round']: re.get('eval1_hit') for re in round_evaluations},
        'eval2_perfect_matches': {re['round']: re.get('eval2_perfect_match') for re in round_evaluations},
        'eval1_hits_series': eval1_hits_series,
        'eval2_hits_series': eval2_hits_series,
        'token_usage': simulation_token_totals,
        'round_evaluations': round_evaluations,  # 全ラウンドの評価結果
        'per_round_metrics': per_round_metrics,   # ラウンドごとの評価値（プロット用）
        'knowledge_gap_predictions': missing_predictions,  # 候補者ごとの項目別欠損予測
        'knowledge_gap_metrics_overall': knowledge_gap_metrics_overall,
        'knowledge_gap_metrics_detail': kg_overall
    }
    
    result_file = results_dir / f'interview_result_sim{simulation_num}_{timestamp}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"結果を保存しました: {result_file}")
    print(f"{'='*60}\n")

    # wandbへのログ出力：ループ中はステップログなし。ここでは配列全体をまとめて保存して終了。
    if wandb_run:
        try:
            _safe_wandb_log(
                wandb_run,
                {
                    "eval1/hits": eval1_hits_series,
                    "eval2/hits": eval2_hits_series,
                    "eval3/accuracy_overall": knowledge_gap_metrics_overall.get("accuracy"),
                    "eval3/f1_overall": knowledge_gap_metrics_overall.get("f1_score"),
                    "eval3/precision_overall": knowledge_gap_metrics_overall.get("precision"),
                    "eval3/recall_overall": knowledge_gap_metrics_overall.get("recall"),
                },
                step=max_rounds + 1,
            )
        except Exception as e:
            print(f"警告: wandbへの記録に失敗しました: {e}")
        finally:
            try:
                wandb_run.finish()
            except Exception:
                pass

    return result_data



    
    
def run_interviews(num_simulations=1, set_index=None, interviewer_model_type=None, interviewer_model_name=None, max_rounds=None, api_provider=None):
    """複数回の面接シミュレーションを実行（簡素版）"""
    print("\n" + "="*80)
    print(f"面接ロールプレイ実行システム - {num_simulations}回のシミュレーション")
    print("="*80)

    if max_rounds is None:
        max_rounds = MAX_ROUNDS
    if interviewer_model_type is None:
        interviewer_model_type = INTERVIEWER_MODEL_TYPE
    if interviewer_model_name is None:
        interviewer_model_name = LOCAL_MODEL_NAME if interviewer_model_type == 'local' else INTERVIEWER_MODEL

    print(f"面接官モデル: {interviewer_model_type} ({interviewer_model_name})")

    experiment_id = generate_experiment_id(set_index, interviewer_model_type, interviewer_model_name, max_rounds)

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    shared_local_model = None
    shared_local_tokenizer = None

    if interviewer_model_type == 'local':
        print(f"\n--- ローカルモデル ({interviewer_model_name}) を初期化します（全シミュレーションで再利用） ---")
        shared_local_model, shared_local_tokenizer = initialize_local_model(interviewer_model_name)
        if shared_local_model is None or shared_local_tokenizer is None:
            print("ローカルモデルの初期化に失敗しました。")
            return
        print(f"--- ローカルモデルの初期化完了 ---\n")

    for sim_num in range(1, num_simulations + 1):
        print(f"\n{'='*80}")
        print(f"シミュレーション {sim_num}/{num_simulations} を実行中...")
        print(f"{'='*80}")

        result = run_single_interview(
            set_index=set_index,
            simulation_num=sim_num,
            interviewer_model_type=interviewer_model_type,
            interviewer_model_name=interviewer_model_name,
            max_rounds=max_rounds,
            local_model=shared_local_model,
            local_tokenizer=shared_local_tokenizer,
            api_provider=api_provider,
            experiment_id=experiment_id,
            wandb_group=experiment_id
        )

        if result:
            all_results.append(result)

    if not all_results:
        print("シミュレーション結果が生成されませんでした。")
        return

    existing_summary_file = find_existing_summary_file(results_dir, experiment_id)

    if existing_summary_file:
        print(f"\n既存のサマリーファイルを発見: {existing_summary_file.name}")
        print(f"実験ID: {experiment_id}")
        print("既存の結果に追加します...")
        try:
            with open(existing_summary_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_individual_results = existing_data.get('individual_results', [])
            existing_individual_results.extend(all_results)
            existing_summary = existing_data.get('experiment_summary', {})
            existing_total = existing_summary.get('total_simulations', 0)
            new_total = existing_total + num_simulations
            summary_data = {
                'experiment_summary': {
                    'total_simulations': new_total,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'created_timestamp': existing_summary.get('created_timestamp', existing_summary.get('timestamp')),
                    'set_index': set_index,
                    'interviewer_model_type': interviewer_model_type,
                    'interviewer_model_name': interviewer_model_name,
                    'max_rounds': max_rounds,
                    'experiment_id': experiment_id
                },
                'individual_results': existing_individual_results
            }
            with open(existing_summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            print(f"\n{'='*80}")
            print("既存のサマリーファイルを更新しました")
            print(f"追加した個別結果: {len(all_results)}件")
            print(f"合計個別結果: {len(existing_individual_results)}件")
            print(f"合計シミュレーション数: {new_total}件")
            print(f"サマリーファイル: {existing_summary_file}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"警告: 既存ファイルの読み込みに失敗しました: {e}")
            print("新規ファイルとして保存します...")
            existing_summary_file = None

    if not existing_summary_file:
        summary_data = {
            'experiment_summary': {
                'total_simulations': num_simulations,
                'timestamp': datetime.datetime.now().isoformat(),
                'created_timestamp': datetime.datetime.now().isoformat(),
                'set_index': set_index,
                'interviewer_model_type': interviewer_model_type,
                'interviewer_model_name': interviewer_model_name,
                'max_rounds': max_rounds,
                'experiment_id': experiment_id
            },
            'individual_results': all_results
        }
        summary_file = results_dir / f'experiment_summary_{timestamp_str}_{experiment_id}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*80}")
        print("全シミュレーション完了")
        print(f"実験ID: {experiment_id}")
        print(f"個別結果: {len(all_results)}件")
        print(f"サマリーファイル: {summary_file}")
        print(f"{'='*80}\n")
