from src.env.state_data import StateData
from src.env.action_data import ActionData
from src.agents.base_agent import BaseAgent
from src.agents.llm_agent.prompt_generator import PromptGenerator, SYSTEM_PROMPT


class LLMAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt_generator = PromptGenerator()

    def select_action(self, state: StateData) -> ActionData:
        """状態stateに基づき、行動データActionDataを返す"""
        prompt = self.prompt_generator.generate_instruction_prompt(state=state)
        commands = state.command_request.commands

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        action = ActionData()

        return action

    def update(self, state, action_data, next_state) -> dict | None:
        pass


    def _llm_inference(self, messages: list) -> int:
        