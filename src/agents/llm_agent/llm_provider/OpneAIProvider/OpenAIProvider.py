from openai import OpenAI
from src.agents.llm_agent.llm_provider.BaseLlmProvider import BaseLlmProvider
from src.agents.llm_agent.llm_provider.OpneAIProvider.api import (
    ApiType,
    API_TYPE_TO_API_KEY,
    API_TYPE_TO_MODEL_NAME,
    API_TYPE_TO_BASE_URL,
)


class OpenAIProvider(BaseLlmProvider):
    def __init__(
        self,
        api_type: ApiType,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ):
        self.client = OpenAI(
            api_key=API_TYPE_TO_API_KEY[api_type],
            base_url=API_TYPE_TO_BASE_URL[api_type],
        )
        self.model = API_TYPE_TO_MODEL_NAME[api_type]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate_response(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
