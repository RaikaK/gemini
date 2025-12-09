from utils import call_openai_api
import sys

print("Testing call_openai_api...")
# Minimal prompt to check connectivity
response, token_info = call_openai_api("gpt-4o-mini", "Hello, are you there? Reply with YES.")

if "API呼び出しエラー" in response or "エラー" in response:
    print(f"Test Failed: {response}")
    sys.exit(1)
else:
    print(f"Response: {response}")
    print(f"Token Info: {token_info}")
    print("Test Passed: call_openai_api executed successfully.")
