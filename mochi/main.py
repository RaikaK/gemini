# main.py - メインの実行スクリプト

import json
import datetime
import random
import argparse
import sys
from pathlib import Path

from config import (
    INTERVIEWER_MODEL, APPLICANT_MODEL, NUM_CANDIDATES, MAX_ROUNDS, 
    ASPIRATION_LEVEL_MAPPING, DB_FILE_PATH, NUM_SIMULATIONS, ENABLE_SPREADSHEET,
    INTERVIEWER_MODEL_TYPE, LOCAL_MODEL_NAME, AVAILABLE_LOCAL_MODELS
)
from interviewer import Interviewer
from student import CompanyKnowledgeManager, Applicant
import re

# スプレッドシート連携のインポート（オプション）
try:
    from spreadsheet_integration import get_spreadsheet_integration
    SPREADSHEET_AVAILABLE = True
except ImportError:
    SPREADSHEET_AVAILABLE = False
    print("警告: spreadsheet_integrationモジュールが見つかりません。スプレッドシート連携は無効です。")

# ローカルモデル管理のインポート（オプション）
try:
    from model_manager import HuggingFaceModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    print("警告: model_managerモジュールが見つかりません。ローカルモデルは使用できません。")

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
        
        # セットインデックスが指定されていない場合はランダムに選択
        if set_index is None:
            set_index = random.randint(0, len(data) - 1)
        elif set_index >= len(data):
            print(f"警告: 指定されたインデックス {set_index} が範囲外です")
            set_index = random.randint(0, len(data) - 1)
        
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


def calculate_ranking_accuracy(candidate_states, ranking_eval):
    """
    ランキング評価の精度指標を計算（改良版）
    正しくランキングが抽出できた場合のみスコアを計算する
    """
    try:
        # 真の志望度ランキングを作成（低い順）
        true_ranking = []
        candidate_names = []
        for i, state in enumerate(candidate_states):
            profile = state['profile']
            preparation = profile.get('preparation', 'low')
            preparation_levels = {'low': 1, 'medium': 2, 'high': 3}
            motivation_score = preparation_levels.get(preparation, 1)
            candidate_name = profile.get('name', f'Candidate_{i+1}')
            true_ranking.append({
                'name': candidate_name,
                'score': motivation_score,
                'preparation': preparation
            })
            candidate_names.append(candidate_name)
        
        # 真のランキングをスコア順にソート（低い順）
        true_ranking.sort(key=lambda x: x['score'])
        true_names = [item['name'] for item in true_ranking]
        
        # 予測ランキングを抽出（より柔軟なパターンに対応）
        predicted_ranking = []
        if isinstance(ranking_eval, str):
            # より柔軟なパターンで抽出を試みる
            # パターン1: "1位: [氏名]" または "1位：[氏名]"
            # パターン2: "**1 位:** [氏名]" または "**1位:** [氏名]"
            # パターン3: "1. [氏名]" または "1.[氏名]"
            # パターン4: 行頭から順位番号と氏名を抽出
            
            extracted_names = {}
            
            # まず、全体テキストに対してマークダウン形式のパターンを試す
            # パターン1: "**数字 位:** 名前" または "**数字位:** 名前"
            markdown_pattern1 = r'\*{2,}\s*(\d+)\s*位\s*\*{0,}\s*[:：]\s*([^\n\*]+?)(?:\n|$|\*|理由)'
            for match in re.finditer(markdown_pattern1, ranking_eval):
                rank_num = int(match.group(1))
                name = match.group(2).strip()
                # 括弧内の名前があれば抽出
                bracket_match = re.search(r'\(([^)]+)\)', name)
                if bracket_match:
                    name = bracket_match.group(1).strip()
                # 候補者Xという形式を除去
                name = re.sub(r'^候補者\d+\s*', '', name)
                name = name.strip()
                if name and rank_num >= 1 and rank_num <= len(candidate_states):
                    extracted_names[rank_num] = name
            
            # パターン2: "**数字. 名前 (説明)**" 形式（例: "**1. 学生CB (志望度の低い順に第1位)**"）
            markdown_pattern2 = r'\*{2,}\s*(\d+)\.\s*([^\n\(\)\*]+?)(?:\s*\([^)]*\))?\s*\*{0,}'
            for match in re.finditer(markdown_pattern2, ranking_eval):
                rank_num = int(match.group(1))
                name = match.group(2).strip()
                # 余分な空白や特殊文字を除去
                name = re.sub(r'^\s+|\s+$', '', name)
                # 候補者Xという形式を除去
                name = re.sub(r'^候補者\d+\s*', '', name)
                name = name.strip()
                if name and rank_num >= 1 and rank_num <= len(candidate_states):
                    extracted_names[rank_num] = name
            
            # 行単位でも処理（マークダウン形式で抽出できなかった場合のフォールバック）
            lines = ranking_eval.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # パターン1: "1位: [氏名]" または "1位: 候補者X (氏名)" 形式
                # 括弧内の名前を優先的に抽出
                match1 = re.search(r'(\d+)位\s*[:：]\s*(?:候補者\d+\s*)?(?:\(([^)]+)\)|([^\s(]+))', line)
                if match1:
                    rank_num = int(match1.group(1))
                    # 括弧内の名前があればそれを使用、なければ括弧前の名前を使用
                    name = match1.group(2) or match1.group(3)
                    if name:
                        name = name.strip()
                        # 余分な括弧や特殊文字を除去
                        name = re.sub(r'[()（）、,，。]', '', name)
                        if name and rank_num >= 1 and rank_num <= len(candidate_states):
                            extracted_names[rank_num] = name
                            continue
                
                # パターン2: "**1 位:** [氏名]" または "**1位:** [氏名]" 形式
                # よりシンプルで確実なパターン
                match2 = re.search(r'\*{2,}\s*(\d+)\s*位\s*\*{0,}\s*[:：]\s*([^\n\*]+?)(?:\s|$|\n|理由)', line)
                if match2:
                    rank_num = int(match2.group(1))
                    name = match2.group(2).strip()
                    # 括弧内の名前があれば抽出（例: "候補者1 (学生BP)" → "学生BP"）
                    bracket_match = re.search(r'\(([^)]+)\)', name)
                    if bracket_match:
                        name = bracket_match.group(1).strip()
                    # 候補者Xという形式を除去
                    name = re.sub(r'^候補者\d+\s*', '', name)
                    # 余分な空白を除去
                    name = name.strip()
                    if name and rank_num >= 1 and rank_num <= len(candidate_states):
                        extracted_names[rank_num] = name
                        continue
                
                # パターン3: "**数字. 名前 (説明)**" 形式（マークダウン形式）
                match3 = re.search(r'\*{2,}\s*(\d+)\.\s*([^\n\(\)\*]+?)(?:\s*\([^)]*\))?\s*\*{0,}', line)
                if match3:
                    rank_num = int(match3.group(1))
                    name = match3.group(2).strip()
                    # 余分な空白を除去
                    name = re.sub(r'^\s+|\s+$', '', name)
                    # 候補者Xという形式を除去
                    name = re.sub(r'^候補者\d+\s*', '', name)
                    if name and rank_num >= 1 and rank_num <= len(candidate_states):
                        extracted_names[rank_num] = name
                        continue
                
                # パターン4: "1. [氏名]" 形式（通常の番号付きリスト）
                match4 = re.search(r'^(\d+)\.\s*(?:候補者\d+\s*)?(?:\(([^)]+)\)|([^\s(]+))', line)
                if match4:
                    rank_num = int(match4.group(1))
                    name = match4.group(2) or match4.group(3)
                    if name:
                        name = name.strip()
                        name = re.sub(r'[()（）、,，。]', '', name)
                        if name and rank_num >= 1 and rank_num <= len(candidate_states):
                            extracted_names[rank_num] = name
                            continue
                
                # パターン5: 行頭の数字とその後の名前
                match5 = re.search(r'^(\d+)\s+(?:候補者\d+\s*)?(?:\(([^)]+)\)|([^\s(]+))', line)
                if match5:
                    rank_num = int(match5.group(1))
                    name = match5.group(2) or match5.group(3)
                    if name:
                        name = name.strip()
                        name = re.sub(r'[()（）、,，。]', '', name)
                        if name and rank_num >= 1 and rank_num <= len(candidate_states):
                            extracted_names[rank_num] = name
            
            # 抽出した名前を順位順に並べる
            for i in range(1, len(candidate_states) + 1):
                if i in extracted_names:
                    predicted_ranking.append(extracted_names[i])
                else:
                    predicted_ranking.append("不明")
            
            # デバッグ: 抽出結果を表示（開発時のみ）
            if "不明" in predicted_ranking:
                print(f"デバッグ: ランキング抽出結果 - extracted_names={extracted_names}, predicted_ranking={predicted_ranking}")
                print(f"デバッグ: 元のテキスト（最初の500文字）: {ranking_eval[:500]}")
        else:
            predicted_ranking = ["不明"] * len(candidate_states)
        
        # 候補者名のマッチング（部分一致も許容）
        def normalize_name(name):
            """名前を正規化（空白、特殊文字を除去）"""
            if not name or name == "不明":
                return None
            # 空白、括弧、特殊文字を除去
            normalized = re.sub(r'[\s()（）、,，。*]', '', name)
            return normalized if normalized else None
        
        # 正規化された候補者名リスト
        normalized_candidate_names = {normalize_name(name): name for name in candidate_names if normalize_name(name)}
        
        # 予測ランキングを正規化してマッチング
        matched_ranking = []
        for pred_name in predicted_ranking:
            normalized_pred = normalize_name(pred_name)
            if normalized_pred and normalized_pred in normalized_candidate_names:
                matched_ranking.append(normalized_candidate_names[normalized_pred])
            else:
                # 部分一致でマッチングを試みる
                matched = False
                for norm_cand_name, orig_cand_name in normalized_candidate_names.items():
                    if normalized_pred and (normalized_pred in norm_cand_name or norm_cand_name in normalized_pred):
                        matched_ranking.append(orig_cand_name)
                        matched = True
                        break
                if not matched:
                    matched_ranking.append("不明")
        
        # 全ての候補者名が正しく抽出できているか検証
        is_valid = True
        if "不明" in matched_ranking:
            is_valid = False
        
        # 重複チェック
        if len(set(matched_ranking)) != len(matched_ranking):
            is_valid = False
        
        # 正しく抽出できていない場合はスコアを計算しない
        if not is_valid:
            return {
                'accuracy': None,  # スコアを計算しない
                'is_valid': False,
                'true_ranking': true_ranking,
                'predicted_ranking': matched_ranking,
                'raw_predicted_ranking': predicted_ranking,
                'message': 'ランキングが正しく抽出できませんでした。スコアは計算されません。'
            }
        
        # 正しく抽出できた場合のみスコアを計算
        # ペアの順位一致率（Concordant Pair Ratio）を計算
        total_pairs = 0
        correct_pairs = 0
        n = len(true_names)
        
        # すべての可能なペア(i, j)を比較 (i < j)
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                
                # 真の順位: true_names[i] の方が true_names[j] よりも志望度が低い
                # 予測順位におけるそれぞれの候補者のインデックスを取得
                try:
                    pred_idx_i = matched_ranking.index(true_names[i])
                    pred_idx_j = matched_ranking.index(true_names[j])
                except ValueError:
                    # これは発生しないはず（is_validでチェック済み）
                    continue
                
                # 順位が一致するかチェック
                if pred_idx_i < pred_idx_j:
                    correct_pairs += 1
        
        ranking_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
        correct_positions = sum(1 for true, pred in zip(true_names, matched_ranking) if true == pred)
        
        return {
            'accuracy': ranking_accuracy,
            'is_valid': True,
            'true_ranking': true_ranking,
            'predicted_ranking': matched_ranking,
            'raw_predicted_ranking': predicted_ranking,
            'correct_pairs': correct_pairs,
            'total_pairs': total_pairs,
            'correct_positions': correct_positions,
            'total_positions': len(true_names)
        }
        
    except Exception as e:
        print(f"ランキング精度の計算中にエラーが発生しました: {e}")
        return {
            'accuracy': None,
            'is_valid': False,
            'error': str(e),
            'message': f'ランキング精度の計算中にエラーが発生しました: {e}'
        }


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


def run_single_interview(set_index=None, simulation_num=1, interviewer_model_type=None, interviewer_model_name=None):
    """単一の面接シミュレーションを実行"""
    print("\n" + "="*60)
    print(f"面接ロールプレイ実行システム - シミュレーション {simulation_num}")
    print("="*60)
    
    # データ読み込み
    company_profile, candidate_profiles, actual_set_index = load_data_from_db(set_index)
    
    if company_profile is None or candidate_profiles is None:
        print("データ読み込みに失敗しました")
        return None
    
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
        local_model, local_tokenizer = initialize_local_model(interviewer_model_name)
        if local_model is None or local_tokenizer is None:
            print("ローカルモデルの初期化に失敗しました。")
            return None
        interviewer = Interviewer(
            company_profile,
            model_type='local',
            model=local_model,
            tokenizer=local_tokenizer
        )
        print(f"--- 面接官タイプ: ローカルモデル ({interviewer_model_name}) ---")
    else:
        interviewer = Interviewer(
            company_profile,
            model_name=interviewer_model_name,
            model_type='api'
        )
        print(f"--- 面接官タイプ: APIモデル ({interviewer_model_name}) ---")
    
    # 応募者を初期化
    applicant = Applicant(APPLICANT_MODEL)
    knowledge_manager = CompanyKnowledgeManager(company_profile)
    
    # 候補者の状態を初期化
    candidate_states = []
    for profile in candidate_profiles:
        knowledge_dict, coverage_str = knowledge_manager.get_knowledge_for_level(
            profile.get('preparation', 'low')
        )
        candidate_states.append({
            'profile': profile,
            'knowledge': knowledge_dict,
            'knowledge_coverage': coverage_str,
            'conversation_log': []
        })
        print(f"\n候補者: {profile['name']} (準備レベル: {profile['preparation']}, 知識カバレッジ: {coverage_str})")
    
    # 面接ラウンドを実行
    asked_questions = []
    
    print(f"\n{'='*60}")
    print(f"面接開始（{MAX_ROUNDS}ラウンド）")
    print(f"{'='*60}\n")
    
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n--- ラウンド {round_num} ---")
        
        # 質問を生成
        question, _ = interviewer.ask_question([], asked_questions)
        asked_questions.append(question)
        print(f"\n面接官の質問: {question}")
        
        # 各候補者が回答
        for state in candidate_states:
            answer, token_info = applicant.generate_answer(
                state['profile'],
                state['knowledge'],
                state['conversation_log'],
                question
            )
            
            # 会話ログに記録
            state['conversation_log'].append({
                'round': round_num,
                'question': question,
                'answer': answer,
                'token_info': token_info
            })
            
            print(f"\n{state['profile']['name']}: {answer}")
    
    # 最終評価
    print(f"\n{'='*60}")
    print("最終評価")
    print(f"{'='*60}\n")
    
    # 評価1: 最も志望度が低い候補者を選定
    print("【評価1: 最も志望度が低い候補者の選定】")
    least_motivated_eval, _ = interviewer.select_least_motivated_candidate(candidate_states)
    print(f"{least_motivated_eval}\n")
    
    # 評価1と評価2の区切り
    print(f"{'='*60}\n")
    
    # 評価2: ランキング
    print("【評価2: 志望度ランキング（低い順）】")
    ranking_eval, _ = interviewer.rank_candidates_by_motivation(candidate_states)
    print(f"{ranking_eval}\n")
    
    # 評価2: ランキング精度の計算
    ranking_accuracy = calculate_ranking_accuracy(candidate_states, ranking_eval)
    if ranking_accuracy:
        if ranking_accuracy.get('is_valid'):
            print(f"精度スコア: {ranking_accuracy['accuracy']:.3f}")
        else:
            print(f"警告: {ranking_accuracy.get('message', 'ランキングが正しく抽出できませんでした')}")
    
    # 結果を保存
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_data = {
        'simulation_num': simulation_num,
        'timestamp': datetime.datetime.now().isoformat(),
        'set_index': actual_set_index,
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
        'evaluations': {
            'least_motivated': least_motivated_eval,
            'ranking': ranking_eval
        },
        'ranking_accuracy': ranking_accuracy
    }
    
    result_file = results_dir / f'interview_result_sim{simulation_num}_{timestamp}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"結果を保存しました: {result_file}")
    print(f"{'='*60}\n")
    
    return result_data


def run_interviews(num_simulations=1, set_index=None, interviewer_model_type=None, interviewer_model_name=None):
    """複数回の面接シミュレーションを実行"""
    print("\n" + "="*80)
    print(f"面接ロールプレイ実行システム - {num_simulations}回のシミュレーション")
    print("="*80)
    
    # モデル設定の表示
    if interviewer_model_type is None:
        interviewer_model_type = INTERVIEWER_MODEL_TYPE
    if interviewer_model_name is None:
        if interviewer_model_type == 'local':
            interviewer_model_name = LOCAL_MODEL_NAME
        else:
            interviewer_model_name = INTERVIEWER_MODEL
    
    print(f"面接官モデル: {interviewer_model_type} ({interviewer_model_name})")
    
    # スプレッドシート連携の初期化
    spreadsheet_integration = None
    if ENABLE_SPREADSHEET and SPREADSHEET_AVAILABLE:
        try:
            spreadsheet_integration = get_spreadsheet_integration()
            if spreadsheet_integration:
                print("スプレッドシート連携が有効です")
            else:
                print("警告: スプレッドシート連携の設定が見つかりません。連携は無効です。")
        except Exception as e:
            print(f"警告: スプレッドシート連携の初期化に失敗しました: {e}")
    
    # 結果保存用のディレクトリ作成
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []
    
    # 複数回のシミュレーションを実行
    for sim_num in range(1, num_simulations + 1):
        print(f"\n{'='*80}")
        print(f"シミュレーション {sim_num}/{num_simulations} を実行中...")
        print(f"{'='*80}")
        
        result = run_single_interview(
            set_index=set_index, 
            simulation_num=sim_num,
            interviewer_model_type=interviewer_model_type,
            interviewer_model_name=interviewer_model_name
        )
        
        if result:
            all_results.append(result)
            
            # スプレッドシートに記録
            if spreadsheet_integration:
                try:
                    print(f"\nスプレッドシートにシミュレーション {sim_num} の結果を記録中...")
                    record_result = spreadsheet_integration.record_experiment_result(result)
                    if record_result.get('success'):
                        print(f"✓ スプレッドシートへの記録が完了しました (行: {record_result.get('row', 'N/A')})")
                    else:
                        print(f"✗ スプレッドシートへの記録エラー: {record_result.get('message')}")
                except Exception as e:
                    print(f"✗ スプレッドシート記録中にエラーが発生しました: {e}")
    
    # 全体結果の保存
    if all_results:
        summary_data = {
            'experiment_summary': {
                'total_simulations': num_simulations,
                'timestamp': datetime.datetime.now().isoformat(),
                'set_index': set_index
            },
            'individual_results': all_results
        }
        
        summary_file = results_dir / f'experiment_summary_{timestamp_str}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"全シミュレーション完了")
        print(f"個別結果: {len(all_results)}件")
        print(f"サマリーファイル: {summary_file}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='面接ロールプレイ実行システム')
    parser.add_argument(
        '-n', '--num-simulations',
        type=int,
        default=NUM_SIMULATIONS,
        help=f'シミュレーション実行回数 (デフォルト: {NUM_SIMULATIONS})'
    )
    parser.add_argument(
        '-s', '--set-index',
        type=int,
        default=None,
        help='使用するデータセットのインデックス（指定しない場合はランダム）'
    )
    parser.add_argument(
        '-t', '--model-type',
        type=str,
        choices=['api', 'local'],
        default=INTERVIEWER_MODEL_TYPE,
        help=f'面接官モデルタイプ: api または local (デフォルト: {INTERVIEWER_MODEL_TYPE})'
    )
    parser.add_argument(
        '-m', '--model-name',
        type=str,
        default=None,
        help='面接官モデル名（apiの場合はOpenAIモデル名、localの場合はAVAILABLE_LOCAL_MODELSのキー）'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='利用可能なローカルモデル一覧を表示して終了'
    )
    
    args = parser.parse_args()
    
    # ローカルモデル一覧を表示
    if args.list_models:
        if MODEL_MANAGER_AVAILABLE:
            model_manager = HuggingFaceModelManager()
            model_manager.print_model_status()
        else:
            print("ローカルモデル管理モジュールが利用できません")
        sys.exit(0)
    
    # 実行回数の検証
    if args.num_simulations < 1:
        print("エラー: シミュレーション実行回数は1以上である必要があります")
        sys.exit(1)
    
    # モデル名の検証
    if args.model_type == 'local':
        if args.model_name and args.model_name not in AVAILABLE_LOCAL_MODELS:
            print(f"エラー: 未知のローカルモデル '{args.model_name}'")
            print(f"利用可能なモデル: {', '.join(AVAILABLE_LOCAL_MODELS.keys())}")
            print("利用可能なモデル一覧を表示するには: python main.py --list-models")
            sys.exit(1)
    
    # シミュレーション実行
    run_interviews(
        num_simulations=args.num_simulations, 
        set_index=args.set_index,
        interviewer_model_type=args.model_type,
        interviewer_model_name=args.model_name
    )
