"""使用するLLMモデル(HuggingFaceのリポジトリID)"""

from enum import StrEnum


class HuggingFaceModelId(StrEnum):
    Gemma3_12B_IT = "google/gemma-3-12b-it"
    Gemma3_4B_IT = "google/gemma-3-4b-it"
    Swallow = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
