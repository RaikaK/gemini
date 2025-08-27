import json
import random
from utils import call_openai_api, parse_json_from_response
from config import GENERATOR_MODEL_NAME

def load_company_and_candidates_from_db(set_index=None):
    """db.jsonから企業情報と学生プロフィールを読み込む"""
    print("--- db.jsonから企業情報と学生プロフィールを読み込み中 ---")
    
    try:
        with open('db.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print("--- db.jsonの形式が不正です ---")
            return None, None, None
        
        # セットインデックスが指定されていない場合はランダムに選択
        if set_index is None:
            set_index = random.randint(0, len(data) - 1)
        elif set_index >= len(data):
            print(f"--- 指定されたセットインデックス {set_index} が範囲外です。最大値: {len(data) - 1} ---")
            set_index = random.randint(0, len(data) - 1)
        
        selected_set = data[set_index]
        
        if 'company' not in selected_set or 'students' not in selected_set:
            print("--- 選択されたセットに企業情報または学生情報が含まれていません ---")
            return None, None, None
        
        company_profile = selected_set['company']
        candidate_profiles = selected_set['students']
        
        # 学生プロフィールにpreparationフィールドを追加（aspiration_levelに基づいて設定）
        for i, profile in enumerate(candidate_profiles):
            aspiration_level = profile.get('aspiration_level', 'medium_70_percent')
            if 'high_90_percent' in aspiration_level:
                profile['preparation'] = 'high'
            elif 'medium_70_percent' in aspiration_level:
                profile['preparation'] = 'medium'
            else:
                profile['preparation'] = 'low'
        
        print(f"--- セット {set_index + 1} を選択しました ---")
        print(f"--- 企業: {company_profile.get('name', 'N/A')} ---")
        print(f"--- 学生数: {len(candidate_profiles)}人 ---")
        
        return company_profile, candidate_profiles, set_index
        
    except FileNotFoundError:
        print("--- db.jsonファイルが見つかりません ---")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"--- db.jsonのJSON形式が不正です: {e} ---")
        return None, None, None
    except Exception as e:
        print(f"--- db.jsonの読み込み中にエラーが発生しました: {e} ---")
        return None, None, None


def generate_company_profile():
    """架空の企業情報を生成する"""
    print("--- 企業情報の生成を開始 ---")
    prompt = """
    日本の架空の企業1社の情報を生成してください。

    以下のJSON形式で出力してください：

    ```json
    {
      "companies": [
        {
          "name": "企業名",
          "business": "事業内容（50字程度）",
          "revenue": "年商XX億円",
          "employees": "従業員数XX名",
          "founded": "設立XXXX年",
          "location": "本社：XX都XX区",
          "vision": "企業ビジョン（30字程度）",
          "products": "主力製品・サービス（30字程度）",
          "culture": "企業文化（30字程度）",
          "recent_news": "最近のニュース・動向（30字程度）",
          "competitive_advantage": "競合優位性（30字程度）",
          "ceo_message": "CEO・代表メッセージ（30字程度）",
          "expansion_plan": "事業展開計画（30字程度）",
          "awards": "受賞歴・評価（30字程度）",
          "partnerships": "パートナーシップ・提携（30字程度）"
        }
      ]
    }
    ```

    要件：
    - 実在しない架空の企業名を使用
    - 多様な業界（IT、製造業、サービス業など）
    - リアルな規模感（従業員数100-5000名程度）
    - 現代的な日本企業として設定
    """
    response = call_openai_api(GENERATOR_MODEL_NAME, prompt)
    data = parse_json_from_response(response)
    
    if isinstance(data, dict) and "companies" in data and len(data["companies"]) > 0:
        company_data = data["companies"][0]
        print("--- 企業情報の生成完了 ---")
        return company_data
    else:
        print("--- 企業情報の生成に失敗、または予期しない形式です ---")
        return {"error": "Failed to generate company profile correctly", "raw_data": data}


def generate_candidate_profiles(company_profile, num_candidates):
    """企業情報に基づき、複数の学生プロフィールを生成する"""
    print(f"--- {num_candidates}人の学生プロフィールの生成を開始 ---")
    prompt = f"""
    以下の企業情報に興味を持つ、日本の架空の就活生{num_candidates}人の情報を生成してください。

    [企業情報]
    {json.dumps(company_profile, ensure_ascii=False)}

    以下のJSON形式で出力してください：

    ```json
    {{
      "candidates": [
        {{
          "name": "日本人の氏名",
          "university": "大学名学部名",
          "gakuchika": "学生時代に力を入れたこと（100字程度の具体的なエピソード）",
          "interest": "興味のある分野・職種",
          "strength": "強み・能力",
          "preparation": "high",
          "mbti": "MBTIタイプ（例：INTJ、ESFJなど）"
        }}
      ]
    }}
    ```

    要件：
    - 多様な大学・学部（国公立・私立・地方大学含む）
    - 具体的で説得力のあるガクチカエピソード
    - 異なる強みや興味分野を持つ学生
    - リアルな日本の就活生として設定
    - 各候補の "preparation" フィールドは以下の順序で必ず出力してください。
      1人目: "low"
      2人目: "medium"
      3人目: "high"
      この順序以外は絶対に許されません。
    """
    response = call_openai_api(GENERATOR_MODEL_NAME, prompt)
    data = parse_json_from_response(response)
    
    if isinstance(data, dict) and "candidates" in data:
        candidates_data = data["candidates"]
        print("--- 学生プロフィールの生成完了 ---")
        return candidates_data
    else:
        print("--- 学生プロフィールの生成に失敗、または予期しない形式です ---")
        return {"error": "Failed to generate candidate profiles correctly", "raw_data": data}
