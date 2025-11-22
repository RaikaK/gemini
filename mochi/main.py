# main.py - メインの実行スクリプト

import json
import datetime
import random
from pathlib import Path

from config import INTERVIEWER_MODEL, APPLICANT_MODEL, NUM_CANDIDATES, MAX_ROUNDS, ASPIRATION_LEVEL_MAPPING, DB_FILE_PATH
from interviewer import Interviewer
from student import CompanyKnowledgeManager, Applicant

def load_data_from_db(set_index=None):
    """db.jsonからデータを読み込む"""
    try:
        # configから相対パスを取得
        db_path = Path(__file__).parent / DB_FILE_PATH
        
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print("エラー: db.jsonの形式が不正です")
            return None, None
        
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
        
        return company_profile, candidate_profiles[:NUM_CANDIDATES]
        
    except FileNotFoundError:
        print("エラー: db.jsonが見つかりません")
        return None, None
    except Exception as e:
        print(f"エラー: データ読み込み失敗 - {e}")
        return None, None


def run_interview():
    """面接シミュレーションを実行"""
    print("\n" + "="*60)
    print("面接ロールプレイ実行システム（最小限版）")
    print("="*60)
    
    # データ読み込み
    company_profile, candidate_profiles = load_data_from_db()
    
    if company_profile is None or candidate_profiles is None:
        print("データ読み込みに失敗しました")
        return
    
    # 面接官と応募者を初期化
    interviewer = Interviewer(company_profile, INTERVIEWER_MODEL)
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
    
    # 最も志望度が低い候補者を選定
    least_motivated_eval, _ = interviewer.select_least_motivated_candidate(candidate_states)
    print(f"\n【最も志望度が低い候補者の選定】\n{least_motivated_eval}")
    
    # ランキング
    ranking_eval, _ = interviewer.rank_candidates_by_motivation(candidate_states)
    print(f"\n【志望度ランキング（低い順）】\n{ranking_eval}")
    
    # 結果を保存
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_data = {
        'timestamp': datetime.datetime.now().isoformat(),
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
        }
    }
    
    result_file = results_dir / f'interview_result_{timestamp}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"結果を保存しました: {result_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_interview()
