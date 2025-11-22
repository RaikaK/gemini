#!/usr/bin/env python3
# demo.py - デモ実行スクリプト（APIキー不要のドライラン）

"""
このスクリプトは、実際のAPI呼び出しなしでシステムの動作を確認するためのデモです。
本番実行は main.py を使用してください。
"""

import json
from pathlib import Path
import sys

# config をインポート
sys.path.insert(0, str(Path(__file__).parent))
from config import ASPIRATION_LEVEL_MAPPING

def run_demo():
    """デモ実行"""
    print("\n" + "="*60)
    print("面接ロールプレイ実行システム - デモモード")
    print("="*60)
    
    print("\n【注意】このデモでは実際のLLM呼び出しは行いません")
    print("実際の実行には main.py を使用してください\n")
    
    # データ読み込みのデモ
    print("="*60)
    print("ステップ1: データ読み込み")
    print("="*60)
    
    try:
        db_path = Path(__file__).parent.parent / 'experiment_inter' / 'db.json'
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ db.jsonから {len(data)} 件のデータセットを読み込みました")
        
        # サンプルデータを表示
        sample = data[0]
        company = sample['company']
        students = sample['students']
        
        print(f"\n【サンプルデータ】")
        print(f"企業名: {company.get('name', 'N/A')}")
        print(f"事業内容: {company.get('business', 'N/A')}")
        print(f"候補者数: {len(students)}人")
        
        print(f"\n候補者情報:")
        for i, student in enumerate(students[:3], 1):
            aspiration_level = student.get('aspiration_level', 'unknown')
            prep = ASPIRATION_LEVEL_MAPPING.get(aspiration_level, 'low')
            
            # 日本語ラベル
            prep_labels = {
                'high': '志望度高',
                'medium': '志望度中',
                'low': '志望度低'
            }
            prep_label = prep_labels.get(prep, '不明')
            
            print(f"  {i}. {student.get('name', 'N/A')} - {prep} ({prep_label})")
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        return
    
    # 面接フローのデモ
    print(f"\n{'='*60}")
    print("ステップ2: 面接フロー（シミュレーション）")
    print("="*60)
    
    print("\n想定される面接の流れ:")
    print("  1. 面接官が質問を生成")
    print("  2. 3名の候補者がそれぞれ回答")
    print("  3. これを5ラウンド繰り返す")
    print("  4. 最終評価を実施")
    
    print("\n【サンプル質問】")
    sample_questions = [
        "当社のビジョンについてどのように理解していますか？",
        "当社の主力製品についてご存知ですか？",
        "なぜ当社に応募されたのですか？",
        "当社の企業文化についてどう思われますか？",
        "入社後にどのような貢献ができると考えていますか？"
    ]
    
    for i, q in enumerate(sample_questions, 1):
        print(f"  ラウンド{i}: {q}")
    
    # 評価のデモ
    print(f"\n{'='*60}")
    print("ステップ3: 最終評価")
    print("="*60)
    
    print("\n評価1: 最も志望度が低い候補者の選定")
    print("  → 各候補者の回答内容から企業知識の深さと熱意を分析")
    
    print("\n評価2: 志望度ランキング（低い順）")
    print("  → 全候補者を志望度の低い順にランキング")
    
    # 出力のデモ
    print(f"\n{'='*60}")
    print("ステップ4: 結果の保存")
    print("="*60)
    
    print("\n結果は results/ ディレクトリに JSON 形式で保存されます")
    print("ファイル名: interview_result_YYYYMMDD_HHMMSS.json")
    
    print("\n【保存される情報】")
    print("  - 企業プロフィール")
    print("  - 各候補者の会話ログ")
    print("  - 最終評価結果")
    print("  - タイムスタンプ")
    
    # 実行方法の案内
    print(f"\n{'='*60}")
    print("実際に実行するには")
    print("="*60)
    
    print("\n1. 依存ライブラリをインストール:")
    print("   pip install -r requirements.txt")
    
    print("\n2. APIキーを設定:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    
    print("\n3. メインスクリプトを実行:")
    print("   python main.py")
    
    print("\n" + "="*60)
    print("デモ終了")
    print("="*60 + "\n")

if __name__ == '__main__':
    run_demo()
