import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import INTERVIEWER_MODEL, API_PROVIDER
from utils import call_openai_api

def verify_gemini_connection():
    print(f"Testing connection with:")
    print(f"  Provider: {API_PROVIDER}")
    print(f"  Model: {INTERVIEWER_MODEL}")
    
    prompt = "Hello! Are you running on Gemini 2.5 Flash Lite? Please answer with 'Yes, I am Gemini' if you are."
    
    print(f"\nSending prompt: '{prompt}'...")
    
    response, token_info = call_openai_api(INTERVIEWER_MODEL, prompt)
    
    print("\n--- Response ---")
    print(response)
    print("----------------")
    print(f"Token Info: {token_info}")
    
    if "Gemini" in response or "Yes" in response:
        print("\nSUCCESS: Verification successful!")
        return True
    elif "API呼び出しエラー" in response or "エラー" in response:
        print("\nFAILURE: API call failed.")
        return False
    else:
        print("\nWARNING: Response received but content is unexpected.")
        return True

if __name__ == "__main__":
    verify_gemini_connection()
