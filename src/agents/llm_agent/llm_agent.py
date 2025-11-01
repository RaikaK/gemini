from transformers import AutoModelForCausalLM, AutoTokenizer

from src.env.state_data import StateData
from src.env.action_data import ActionData
from src.agents.base_agent import BaseAgent
from src.agents.llm_agent.prompt_generator import PromptGenerator, SYSTEM_PROMPT
from src.agents.llm_agent.llm_provider.BaseLlmProvider import BaseLlmProvider
from src.agents.llm_agent.llm_provider.HuggingFaceProvider import HuggingFaceProvider


class LLMAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt_generator = PromptGenerator()
        # LLM Provider
        self.provider: BaseLlmProvider = HuggingFaceProvider()

    def select_action(self, state: StateData) -> ActionData:
        """状態stateに基づき、行動データActionDataを返す"""
        prompt = self.prompt_generator.generate_instruction_prompt(state=state)
        cmd_request = state.command_request

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response: str = self.provider.generate_response(message=messages)

        cmd_index: int = self._parse_response_to_cmd_index(
            response, cmd_request.commands
        )
        cmd_entry = cmd_request.commands[cmd_index]
        action = ActionData(command_request=cmd_request, command_entry=cmd_entry)

        return action

    def update(self, state, action_data, next_state) -> dict | None:
        pass

    def _parse_response_to_cmd_index(self, response: str, commands: list) -> int:
        pass
