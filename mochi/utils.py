import json
from openai import OpenAI
import google.generativeai as genai
import os
from config import OPENAI_API_KEY, GOOGLE_API_KEY

def call_openai_api(model_name, prompt):
    """
    LLM APIを呼び出してテキスト応答を取得
    model_nameが 'gemini' で始まる場合はGoogle API、それ以外はOpenAI APIを使用
    """
    # Geminiモデルの場合
    if model_name.lower().startswith("gemini"):
        if not GOOGLE_API_KEY or not GOOGLE_API_KEY.strip() or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            error_message = "GOOGLE_API_KEYが設定されていません。環境変数GOOGLE_API_KEYを設定してください。"
            print(f"エラー: {error_message}")
            return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # Geminiはトークン情報を標準では返さないが、簡易的にカウントするか0を入れる
            # (必要であれば count_tokens API を呼ぶことも可能だがスキップ)
            token_info = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            return response.text, token_info
        except Exception as e:
            error_str = str(e)
            print(f"Gemini APIエラー: {error_str}")
            return f"API呼び出しエラー: {error_str}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # OpenAIモデルの場合 (デフォルト)
    # APIキーの検証
    if not OPENAI_API_KEY or not OPENAI_API_KEY.strip() or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        error_message = "APIキーが設定されていません。環境変数OPENAI_API_KEYを設定してください。"
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
        error_str = str(e)
        # 401エラーの場合、より明確なメッセージを表示
        if "401" in error_str or "invalid_api_key" in error_str.lower() or "incorrect api key" in error_str.lower():
            error_message = (
                "APIキーが無効です。以下の点を確認してください:\n"
                "1. 環境変数OPENAI_API_KEYに正しいAPIキーが設定されているか\n"
                "2. APIキーが有効期限内であるか\n"
                "3. APIキーに必要な権限があるか\n"
                f"エラー詳細: {error_str}"
            )
        else:
            error_message = f"API呼び出しエラー: {error_str}"
        print(f"エラー: {error_message}")
        return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
