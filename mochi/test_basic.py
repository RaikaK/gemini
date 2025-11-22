#!/usr/bin/env python3
# test_basic.py - 基本的な機能テスト（API呼び出しなし）

import json
from pathlib import Path

def test_load_data():
    """データ読み込みのテスト"""
    print("テスト1: データ読み込み")
    
    try:
        db_path = Path(__file__).parent.parent / 'experiment_inter' / 'db.json'
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ db.jsonの読み込み成功")
        print(f"  - データセット数: {len(data)}")
        
        if len(data) > 0:
            print(f"  - 1つ目のセット:")
            print(f"    - 企業名: {data[0]['company'].get('name', 'N/A')}")
            print(f"    - 学生数: {len(data[0].get('students', []))}")
        
        return True
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

def test_imports():
    """インポートのテスト"""
    print("\nテスト2: モジュールインポート")
    
    try:
        import config
        print(f"✓ config.py インポート成功")
        
        from utils import call_openai_api
        print(f"✓ utils.py インポート成功")
        
        from interviewer import Interviewer
        print(f"✓ interviewer.py インポート成功")
        
        from student import CompanyKnowledgeManager, Applicant
        print(f"✓ student.py インポート成功")
        
        return True
    except Exception as e:
        print(f"✗ インポートエラー: {e}")
        return False

def test_knowledge_manager():
    """知識マネージャーのテスト"""
    print("\nテスト3: CompanyKnowledgeManager")
    
    try:
        sample_company = {
            "name": "テスト株式会社",
            "business": "ソフトウェア開発",
            "products": "クラウドサービス",
            "vision": "世界一のサービスを作る",
            "culture": "チャレンジ精神",
            "awards": "ベストカンパニー賞"
        }
        
        from student import CompanyKnowledgeManager
        km = CompanyKnowledgeManager(sample_company)
        
        # 各レベルでテスト
        for level in ['low', 'medium', 'high']:
            knowledge, coverage = km.get_knowledge_for_level(level)
            known_count = sum(1 for v in knowledge.values() if v != "")
            print(f"✓ レベル '{level}': {coverage} - {known_count}項目を保持")
        
        return True
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("基本機能テスト")
    print("="*60)
    
    results = []
    results.append(test_load_data())
    results.append(test_imports())
    results.append(test_knowledge_manager())
    
    print("\n" + "="*60)
    if all(results):
        print("全テスト成功 ✓")
    else:
        print(f"テスト結果: {sum(results)}/{len(results)} 成功")
    print("="*60)
