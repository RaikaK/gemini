from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import random

from src.env.state_data import StateData
from src.env.action_data import ActionData
from src.agents.base_agent import BaseAgent
from src.agents.llm_agent.prompt_generator import PromptGenerator, SYSTEM_PROMPT
from src.agents.llm_agent.llm_provider.BaseLlmProvider import BaseLlmProvider
from src.agents.llm_agent.llm_provider.HuggingFaceProvider.HuggingFaceProvider import (
    HuggingFaceProvider,
)
from src.agents.llm_agent.llm_provider.HuggingFaceProvider.hf_models import (
    HuggingFaceModelId,
)


class LLMAgent(BaseAgent):
    def __init__(
        self,
        model_id: HuggingFaceModelId = HuggingFaceModelId.Swallow,
        max_try: int = 3,
    ):
        super().__init__()
        self.prompt_generator = PromptGenerator()
        # LLM Provider
        self.provider: BaseLlmProvider = HuggingFaceProvider(model_id=model_id)
        self.max_try = max_try

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        """状態stateに基づき、行動データActionDataを返す"""
        prompt = self.prompt_generator.generate_instruction_prompt(state=state)
        cmd_request = state.command_request

        # 無駄な選択は避ける
        if len(cmd_request.commands) == 1:
            return ActionData(
                command_request=cmd_request, command_entry=cmd_request.commands[0]
            ), None

        # breakpoint()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response: str = self.provider.generate_response(messages=messages)

        breakpoint()

        can_parsed = False
        n_try = 0
        while not can_parsed:
            n_try += 1
            cmd_index, json_object = self._parse_response_to_cmd_index(
                response, cmd_request.commands
            )
            breakpoint()
            if cmd_index is not None:
                cmd_entry = cmd_request.commands[cmd_index]
                action = ActionData(
                    command_request=cmd_request, command_entry=cmd_entry
                )
                return action, json_object

            if n_try >= self.max_try:
                break

        # ここまで来たら、ランダムで結果を返す
        cmd_entry = random.choice(cmd_request.commands)
        return ActionData(command_entry=cmd_entry, command_request=cmd_request), None

    def update(
        self,
        state: StateData,
        action: ActionData,
        next_state: StateData,
        info: dict | None,
    ) -> dict | None:
        pass

    def _parse_response_to_cmd_index(
        self, response: str, commands: list
    ) -> tuple[int | None, dict]:
        expected_keys = {"reasoning", "action"}
        parsed_json_object: dict | None = self._extract_json_from_text(response)
        breakpoint()
        """commandsの中から選択されたかチェックする"""
        if parsed_json_object is None:
            return None, {}
        if set(parsed_json_object.keys()) != expected_keys:
            return None, {}
        # keyの確認OK
        action_index = parsed_json_object["action"]
        if action_index in [i for i in range(len(commands))]:
            return action_index, parsed_json_object
        return None, {}

    def _extract_json_from_text(self, text: str) -> dict:
        """
        テキストからJSON部分を正規表現で抜き取る
        """
        # パターン1: ```json から ``` までを抜き取る
        json_pattern1 = r"```json\s*(.*?)\s*```"
        match1 = re.search(json_pattern1, text, re.DOTALL | re.IGNORECASE)

        if match1:
            try:
                json_obj = json.loads(match1.group(1))
                return json_obj
            except Exception as e:
                return None
        # パターン2: { から } までの最初のJSONブロックを抜き取る
        json_pattern2 = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match2 = re.search(json_pattern2, text, re.DOTALL)

        if match2:
            try:
                json_obj = json.loads(match2.group(0))
                return json_obj
            except Exception as e:
                return None
        return None
