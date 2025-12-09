# utils.py - ユーティリティ関数

import json
import time
import random
from openai import OpenAI, RateLimitError, APIError
from config import OPENAI_API_KEY

def call_openai_api(model_name, prompt):
    """OpenAI APIを呼び出してテキスト応答を取得（リトライ機能付き）"""
    # APIキーの検証
    if not OPENAI_API_KEY or not OPENAI_API_KEY.strip() or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        error_message = "APIキーが設定されていません。環境変数OPENAI_API_KEYを設定してください。"
        print(f"エラー: {error_message}")
        return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    max_retries = 5
    base_delay = 2  # 初期待機時間（秒）
    
    for attempt in range(max_retries):
        try:
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
            
        except RateLimitError as e:
            # 429 Rate Limit Errors
            if attempt < max_retries - 1:
                # 指数バックオフ + ジッター
                delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                print(f"警告: レート制限(429)が発生しました。{delay:.2f}秒後にリトライします... ({attempt+1}/{max_retries})")
                print(f"詳細: {e}")
                time.sleep(delay)
            else:
                error_message = f"API呼び出しエラー (レート制限超過): {e}"
                print(f"エラー: {error_message}")
                print("ヒント: アカウントのクレジット残高や利用制限(Quota)を確認してください。")
                return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
        except APIError as e:
            # その他のAPIエラー（500系など）
            if e.status_code in [500, 503] and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"警告: サーバーエラー({e.status_code})が発生しました。{delay}秒後にリトライします... ({attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                error_message = f"API呼び出しエラー: {e}"
                print(f"エラー: {error_message}")
                return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
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
                error_message = f"予期せぬエラー: {error_str}"
            
            print(f"エラー: {error_message}")
            return error_message, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
