# utils.py - ユーティリティ関数

import json
from openai import OpenAI
from config import OPENAI_API_KEY

def call_openai_api(model_name, prompt):
    """OpenAI APIを呼び出してテキスト応答を取得"""
    # APIキーの検証
    if not OPENAI_API_KEY or not OPENAI_API_KEY.strip() or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        error_message = "APIキーが設定されていません"
        print(f"エラー: {error_message}")
        return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # トークン数情報を取得
        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
        
        return response.choices[0].message.content, token_info
    except Exception as e:
        error_message = f"API呼び出しエラー: {e}"
        print(f"エラー: {error_message}")
        return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
