#!/usr/bin/env python3
# validate_structure.py - コード構造の検証（ライブラリ不要）

import json
import ast
from pathlib import Path

def validate_file_exists():
    """必要なファイルの存在確認"""
    print("検証1: ファイル構成")
    
    required_files = [
        'config.py',
        'utils.py',
        'interviewer.py',
        'student.py',
        'main.py',
        'README.md',
        'requirements.txt'
    ]
    
    base_dir = Path(__file__).parent
    all_exist = True
    
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} が見つかりません")
            all_exist = False
    
    return all_exist

def validate_db_json():
    """db.jsonの検証"""
    print("\n検証2: db.jsonの読み込み")
    
    try:
        db_path = Path(__file__).parent.parent / 'experiment_inter' / 'db.json'
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ db.json読み込み成功")
        print(f"  データセット数: {len(data)}")
        
        # 最初のデータセットの構造を確認
        if len(data) > 0:
            first_set = data[0]
            has_company = 'company' in first_set
            has_students = 'students' in first_set
            
            if has_company and has_students:
                print(f"✓ データ構造が正しい")
                print(f"  企業: {first_set['company'].get('name', 'N/A')}")
                print(f"  学生: {len(first_set['students'])}人")
                return True
            else:
                print(f"✗ データ構造が不正")
                return False
        
        return True
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

def validate_python_syntax():
    """Pythonコードの構文チェック"""
    print("\n検証3: Pythonコード構文")
    
    py_files = [
        'config.py',
        'utils.py',
        'interviewer.py',
        'student.py',
        'main.py'
    ]
    
    base_dir = Path(__file__).parent
    all_valid = True
    
    for filename in py_files:
        filepath = base_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
            print(f"✓ {filename} - 構文OK")
        except SyntaxError as e:
            print(f"✗ {filename} - 構文エラー: {e}")
            all_valid = False
        except Exception as e:
            print(f"✗ {filename} - エラー: {e}")
            all_valid = False
    
    return all_valid

def validate_config():
    """設定ファイルの検証"""
    print("\n検証4: 設定ファイル")
    
    try:
        config_path = Path(__file__).parent / 'config.py'
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_vars = [
            'OPENAI_API_KEY',
            'INTERVIEWER_MODEL',
            'APPLICANT_MODEL',
            'NUM_CANDIDATES',
            'MAX_ROUNDS'
        ]
        
        all_found = True
        for var in required_vars:
            if var in content:
                print(f"✓ {var} 定義あり")
            else:
                print(f"✗ {var} 定義なし")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

def validate_readme():
    """READMEの検証"""
    print("\n検証5: README.md")
    
    try:
        readme_path = Path(__file__).parent / 'README.md'
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_sections = [
            '概要',
            'セットアップ',
            '使い方',
            'ファイル構成'
        ]
        
        all_found = True
        for section in required_sections:
            if section in content:
                print(f"✓ '{section}' セクションあり")
            else:
                print(f"✗ '{section}' セクションなし")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("コード構造検証")
    print("="*60 + "\n")
    
    results = []
    results.append(validate_file_exists())
    results.append(validate_db_json())
    results.append(validate_python_syntax())
    results.append(validate_config())
    results.append(validate_readme())
    
    print("\n" + "="*60)
    if all(results):
        print("✓ 全ての検証に成功しました")
        print("\n次のステップ:")
        print("1. pip install -r requirements.txt を実行")
        print("2. config.py のAPIキーを設定")
        print("3. python main.py を実行")
    else:
        print(f"検証結果: {sum(results)}/{len(results)} 成功")
        print("いくつかの項目で問題が見つかりました")
    print("="*60)
