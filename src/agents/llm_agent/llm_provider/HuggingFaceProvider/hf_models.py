"""使用するLLMモデル(HuggingFaceのリポジトリID)"""

from enum import StrEnum


class HuggingFaceModelId(StrEnum):
    Gemma3 = "google/gemma-3-1b-it"
    Swallow = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
