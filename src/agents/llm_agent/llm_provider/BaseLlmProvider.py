import abc


class BaseLlmProvider(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_response(self, messages: list[dict]) -> str:
        pass
