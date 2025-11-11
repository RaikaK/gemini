from enum import Enum, auto


class ApiType(Enum):
    GeminiApi = auto()
    OllamaApi = auto()


API_TYPE_TO_API_KEY = {
    ApiType.GeminiApi: "AIzaSyDq6JjUego6YL5rflXfFGbwf-QNz2O4yW0",
    ApiType.OllamaApi: "ollama-api-key",
}

API_TYPE_TO_BASE_URL: dict[ApiType, str] = {
    ApiType.GeminiApi: "https://generativelanguage.googleapis.com/v1beta/openai/",
    ApiType.OllamaApi: "http://localhost:11434/v1",
}


API_TYPE_TO_MODEL_NAME: dict[ApiType, str] = {
    ApiType.GeminiApi: "gemini-2.0-flash-lite",
    ApiType.OllamaApi: "gemma3:12b",
}
