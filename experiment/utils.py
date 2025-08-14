# エラー処理などの細かい処理の関数

import json
import google.generativeai as genai
from config import GEMINI_API_KEY

def call_gemini_api(model_name, prompt):
    """Gemini APIを呼び出し、テキスト応答を返す"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        error_message = "APIキーが設定されていません。処理を中断します。"
        print(f"エラー: {error_message}")
        if "JSON" in prompt:
            return json.dumps({"error": error_message})
        return error_message
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API呼び出しエラー: {e}"

def parse_json_from_response(response_text):
    """LLMの応答からJSON部分を抽出してパースする"""
    try:
        if '```json' in response_text:
            json_string = response_text.split('```json')[1].split('```')[0]
        else:
            json_string = response_text
        return json.loads(json_string.strip())
    except (json.JSONDecodeError, IndexError) as e:
        print(f"JSONのパースに失敗しました。エラー: {e}")
        return {"error": "Failed to parse JSON", "raw_output": response_text}
