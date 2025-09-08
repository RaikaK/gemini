# エラー処理などの細かい処理の関数

import json
from openai import OpenAI
from config import OPENAI_API_KEY

def call_openai_api(model_name, prompt):
    """OpenAI APIを呼び出し、テキスト応答とtoken数情報を返す"""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        error_message = "APIキーが設定されていません。処理を中断します。"
        print(f"エラー: {error_message}")
        if "JSON" in prompt:
            return json.dumps({"error": error_message}), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # token数情報を取得
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
        
        return response.choices[0].message.content, token_info
    except Exception as e:
        error_message = f"API呼び出しエラー: {e}"
        return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# 後方互換性のためのエイリアス
def call_gemini_api(model_name, prompt):
    """後方互換性のため、call_openai_apiを呼び出す"""
    return call_openai_api(model_name, prompt)

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
