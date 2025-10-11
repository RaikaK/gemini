import torch
import numpy as np
import datatime
import os
import copy

from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from src.common.sample_tensor import (
    BOARD_NUM,
    INFO_NUM,
    ACTION_NUM,
    DNN_INPUT_NUM,
    set_board_vector,
    set_action_vector,
    create_input_data,
)
from src.agents.base_agent import BaseAgent
from src.env.state_data import StateData
from src.env.action_data import ActionData
from src.agents.ppo_agent.actor_critic_model import ActorCriticDnn
from src.agents.ppo_agent.ppo_utils import RolloutBuffer, compute_returns_and_advantages


class PPOAgent(BaseAgent):
    def __init__(
        self,
        epochs_on_update: int = 16,  # 経験データを学習する回数
        batch_size: int = 64,  # ミニバッチサイズ
        clip_epsilon: float = 0.2,  # クリッピングの範囲
        gamma: float = 0.9,  # 割引率
        lambda_gae: float = 0.95,  # GAEのλパラメータ
        lr: float = 1e-5,
    ):
        super().__init__()
        self.epochs_on_update = epochs_on_update
        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.rollout_buffer = RolloutBuffer()  # 1episodeごとにクリアする

        # nn
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.lr = lr
        self.input_size = DNN_INPUT_NUM
        self.pi = ActorCriticDnn(input_size=self.input_size)
        self.old_pi = copy.deepcopy(self.pi)
        self.pi.to(self.device)
        self.optimizer = torch.optim.SGD(self.actor_critic.parameters(), lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        # 入力テンソル
        x_tensor = self._create_input_tensor(state).to(self.device)
        # 推論モードにして、アクションの確率分布を生成
        self.pi.eval()
        action_values, state_value = self.pi(x_tensor)
        probs = torch.softmax(action_values)
        breakpoint()  # shapeとcmd_countの確認を行う

        action_index = np.random.randint(0, len(probs))
        breakpoint()  # action_indexのshapeの確認
        action_prob = probs[action_index]
        action_prob = action_prob.unsqueeze(0).detach().cpu().numpy()

        breakpoint()  # 選択された確率値を確認

        action = ActionData(
            command_request=state.command_request,
            command_entry=state.command_request.commands[action_index],
        )

        info = {"action_prob": action_prob, "state_value": state_value}
        return action, info

    def update(
        self, state: StateData, action: ActionData, next_state: StateData, info: dict
    ) -> dict | None:
        # rollout_bufferに追加
        self.rollout_buffer.add(
            state=state,
            action=action,
            reward=next_state.reward,
            done=next_state.is_duel_end,
            next_state=next_state,
            action_prob=info["aciton_prob"],
            state_value=info["state_value"],
        )

        is_duel_end = next_state.is_duel_end
        if not is_duel_end:
            return None

        # episode終了時に学習させる。
        data = self.rollout_buffer.rollout()
        self.rollout_buffer.clear()
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        next_states = data["next_states"]
        action_probs = data["action_probs"]
        state_values = data["state_values"]
        for epoch in range(self.epochs_on_update):
            pi_old = action_probs
            pi = [
                self._get_current_policy(s, a, is_train=True)
                for s, a in zip(state, actions)
            ]
            ratio_unclipped = [p / (p_old + 1e-10) for p, p_old in zip(pi, pi_old)]
            ratio_clip = [torch.clamp(r, 1 - self.clip_epsilon, 1 + self.clip_epsilon) for r in ratio_unclipped]
            loss_clip = 

    def _create_input_tensor(self, state: StateData) -> torch.Tensor:
        """状態s, 行動aに対する現在の方策pi(s, a)を取得する"""
        input_data = create_input_data(state)
        board_vector = set_board_vector(input_data)
        action_vector = set_action_vector(input_data)
        command_request = state.command_request
        cmd_count = len(command_request.commands)
        # テンソルの作成
        x = np.empty((cmd_count, self.input_size), dtype=np.float32)
        for i in range(cmd_count):
            x[i][0 : BOARD_NUM + INFO_NUM] = board_vector
            x[i][BOARD_NUM + INFO_NUM : self.input_size] = action_vector[i]
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x_tensor

    def _get_current_policy(
        self, state: StateData, action: ActionData, is_train: bool = False
    ) -> torch.Tensor:
        """状態s, 行動aに対する現在の方策pi(s, a)を取得する"""
        input_tensor = self._create_input_tensor(state).to(self.device)
        if is_train:
            self.pi.train()
        else:
            self.pi.eval()

        action_values, _ = self.pi(input_tensor)
        action_prob = torch.softmax(action_values, dim=-1)[action.command_index]
        return action_prob
