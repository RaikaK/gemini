# utils.py - ユーティリティ関数

import json
from openai import OpenAI
from config import OPENAI_API_KEY, GOOGLE_API_KEY, API_PROVIDER

def call_openai_api(model_name, prompt):
    """APIを呼び出してテキスト応答を取得 (OpenAI / Google)"""
    
    api_key = None
    base_url = None
    
    # プロバイダーごとの設定
    if API_PROVIDER == "google":
        api_key = GOOGLE_API_KEY
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
             error_message = "Google APIキーが設定されていません。環境変数GOOGLE_API_KEYを設定してください。"
             print(f"エラー: {error_message}")
             return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    else:
        # デフォルトはOpenAI
        api_key = OPENAI_API_KEY
        if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
            error_message = "OpenAI APIキーが設定されていません。環境変数OPENAI_API_KEYを設定してください。"
            print(f"エラー: {error_message}")
            return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    try:
        # クライアントの初期化
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
            
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
        # エラーメッセージの整形
        if "401" in error_str or "invalid_api_key" in error_str.lower() or "incorrect api key" in error_str.lower():
            if API_PROVIDER == "google":
                error_message = (
                    "Google APIキーが無効です。以下の点を確認してください:\n"
                    "1. 環境変数GOOGLE_API_KEYに正しいAPIキーが設定されているか\n"
                    "2. Google AI StudioでAPIキーが作成されているか\n"
                    f"エラー詳細: {error_str}"
                )
            else:
                error_message = (
                    "OpenAI APIキーが無効です。以下の点を確認してください:\n"
                    "1. 環境変数OPENAI_API_KEYに正しいAPIキーが設定されているか\n"
                    "2. APIキーが有効期限内であるか\n"
                    f"エラー詳細: {error_str}"
                )
        else:
            error_message = f"API呼び出しエラー (Provider: {API_PROVIDER}): {error_str}"
        print(f"エラー: {error_message}")
        return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def get_api_client(provider=None):
    """
    設定または指定されたプロバイダーに基づいてAPIクライアントを取得する
    
    Args:
        provider (str, optional): 使用するAPIプロバイダー ('openai' または 'google')。
                                  Noneの場合は config.API_PROVIDER が使用されます。
    
    Returns:
        OpenAI: 初期化されたOpenAIクライアント（または互換クライアント）
    """
    if provider is None:
        from config import API_PROVIDER
        provider = API_PROVIDER
    
    api_key = None
    base_url = None
    
    if provider == "google":
        from config import GOOGLE_API_KEY
        api_key = GOOGLE_API_KEY
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
             print("警告: Google APIキーが設定されていません。")
    else:
        # デフォルトはOpenAI
        from config import OPENAI_API_KEY
        api_key = OPENAI_API_KEY
        if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
            print("警告: OpenAI APIキーが設定されていません。")
    
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenAI(api_key=api_key)
