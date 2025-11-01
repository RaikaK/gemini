from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from src.agents.llm_agent.llm_provider.HuggingFaceProvider.hf_models import (
    HuggingFaceModelId,
)
from src.agents.llm_agent.llm_provider.BaseLlmProvider import BaseLlmProvider


class HuggingFaceProvider(BaseLlmProvider):
    def __init__(self, model_id: HuggingFaceModelId = HuggingFaceModelId.Gemma3):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id.value)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id.value,
            dtype=torch.bfloat16,
        ).to(self.device)

    def generate_response(self, messages: list[dict]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        breakpoint()
        return response


if __name__ == "__main__":
    provider = HuggingFaceProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    response = provider.generate_response(messages)
    print("Response:", response)
