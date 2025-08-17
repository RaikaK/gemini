import json
from utils import call_openai_api, parse_json_from_response
from config import GENERATOR_MODEL_NAME

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
    - `preparation`は、{num_candidates}人分を 'high', 'medium', 'low' に1人ずつ割り当ててください。
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
