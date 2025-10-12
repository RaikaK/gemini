import torch
from torch.distributions import Categorical
import numpy as np
import datetime
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
from src.agents.ppo.actor_critic_model import Actor, Critic

# from src.agents.ppo.ppo_utils import compute_returns_and_advantages, calc_entropy
from src.agents.ppo.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    def __init__(
        self,
        epochs_on_update: int = 16,  # 経験データを学習する回数
        clip_epsilon: float = 0.2,  # クリッピングの範囲
        gamma: float = 0.9,  # 割引率
        lambda_gae: float = 0.95,  # GAEのλパラメータ
        lr: float = 1e-5,  # 学習率
        c_mse: float = 0.5,  # 状態価値損失の係数
        c_entropy: float = 0.01,  # エントロピーボーナスの係数
    ):
        super().__init__()
        self.epochs_on_update = epochs_on_update
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.c_mse = c_mse
        self.c_entropy = c_entropy

        self.rollout_buffer = RolloutBuffer()  # 1episodeごとにクリアする

        # nn
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.lr = lr
        self.pi_input_size = DNN_INPUT_NUM
        self.critic_input_size = BOARD_NUM + INFO_NUM  # 状態価値はアクション情報を含まない
        self.pi = Actor(input_size=self.pi_input_size)
        self.old_pi = copy.deepcopy(self.pi)
        self.critic = Critic(input_size=self.critic_input_size)

        self.pi.to(self.device)
        self.critic.to(self.device)

        self.optim_pi = torch.optim.SGD(self.pi.parameters(), lr=self.lr)
        self.optim_critic = torch.optim.SGD(self.critic.parameters(), lr=self.lr)

        self.loss_func = torch.nn.MSELoss()

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        # 入力テンソル
        x_tensor = self._create_input_tensor(state).to(self.device)
        # 推論モードにして、アクションの確率分布を生成
        self.pi.eval()
        with torch.no_grad():
            action_values = self.pi(x_tensor)
            probs = torch.softmax(action_values, dim=-1).flatten()
            probs /= torch.sum(probs)
            # breakpoint()  # shapeとcmd_countの確認を行う

        # 状態価値を一緒に出力
        self.critic.eval()
        with torch.no_grad():
            board_vector = set_board_vector(create_input_data(state))
            board_vector_tensor = torch.tensor(board_vector, dtype=torch.float32).to(self.device)
            state_value = self.critic(board_vector_tensor).cpu().detach().numpy()

        action_index = np.random.choice(len(probs), p=probs.detach().cpu().numpy())
        # breakpoint()  # action_indexのshapeの確認
        action_prob = probs[action_index]
        action_prob = action_prob.unsqueeze(0).detach().cpu().numpy()

        # breakpoint()  # 選択された確率値を確認

        action = ActionData(
            command_request=state.command_request,
            command_entry=state.command_request.commands[action_index],
        )

        info = {"action_prob": action_prob, "state_value": state_value}
        return action, info

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict) -> dict | None:
        # rollout_bufferに追加
        reward = next_state.reward
        self.rollout_buffer.add(
            state=state,
            action=action,
            reward=reward,
            done=next_state.is_duel_end,
            next_state=next_state,
            info=info,
        )

        is_duel_end = next_state.is_duel_end
        if not is_duel_end:
            return None

        # episode終了時に学習させる。
        data = self.rollout_buffer.rollout()
        # self.rollout_buffer.clear()
        states: list[StateData] = data["states"]
        actions: list[ActionData] = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        next_states = data["next_states"]
        action_probs = [torch.tensor(d["action_prob"]).to(self.device) for d in data["infos"]]
        state_values = [torch.tensor(d["state_value"]).to(self.device) for d in data["infos"]]
        returns, advantages = self._compute_returns_and_advantages(
            state_values=state_values,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            lambda_gae=self.lambda_gae,
        )

        # breakpoint()  # advantageとstatesなどの長さを確認

        for epoch in range(self.epochs_on_update):
            # データ収集時の方策
            pi_old = [action_prob.to(self.device) for action_prob in action_probs]
            action_probs_list = [
                self._get_current_policy(s, a, is_train=True) for s, a in zip(states, actions)
            ]  # こいつがcudaにある
            # breakpoint()
            # 現在の方策
            pi = [
                action_prob[action.command_index]
                for i, (action_prob, action) in enumerate(zip(action_probs_list, actions))
            ]
            # breakpoint()s
            # 方策比をクリッピング
            ratio_unclipped = torch.stack(
                [p / (p_old + 1e-10) * advantage for p, p_old, advantage in zip(pi, pi_old, advantages)]
            )
            ratio_clip = torch.stack(
                [
                    torch.clamp(r, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                    for r, advantage in zip(ratio_unclipped, advantages)
                ]
            )

            # エントロピーボーナス
            dist_entropy = [self._calc_entropy(action_prob=action_prob) for action_prob in action_probs_list]
            dist_entropy_bonus = torch.stack(dist_entropy)

            # breakpoint(s)
            loss_pi = (
                -torch.min(ratio_unclipped, ratio_clip)
                + self.c_mse * self.loss_func(torch.stack(state_values), torch.stack(returns))
                - self.c_entropy * dist_entropy_bonus
            ).mean()

            self.optim_pi.zero_grad()
            loss_pi.backward()
            self.optim_pi.step()

            board_tensor = torch.tensor([set_board_vector(create_input_data(state)) for state in states]).to(
                self.device
            )
            # breakpoint()  # board_tensorのshapeとreturnsのshapeを確認
            loss_critic = self.loss_func(self.critic(board_tensor), torch.stack(returns)).mean()
            self.optim_critic.zero_grad()
            loss_critic.backward()
            self.optim_critic.step()

            # losses_pi.append(loss_pi.detach().cpu().numpy().mean())

        return {
            "loss_pi": loss_pi,
            "loss_critic": loss_critic,
            "reward": reward,
            "ave_reward": np.average(rewards) if len(rewards) > 0 else 0,
            "finish_type(1:Normal|2:NoDeck)": next_state.duel_end_data.finish_type,
        }

    def _create_input_tensor(self, state: StateData) -> torch.Tensor:
        """状態s, 行動aに対する現在の方策pi(s, a)を取得する"""
        input_data = create_input_data(state)
        board_vector = set_board_vector(input_data)
        action_vector = set_action_vector(input_data)
        command_request = state.command_request
        cmd_count = len(command_request.commands)
        # テンソルの作成
        x = np.empty((cmd_count, self.pi_input_size), dtype=np.float32)
        for i in range(cmd_count):
            x[i][0 : BOARD_NUM + INFO_NUM] = board_vector
            x[i][BOARD_NUM + INFO_NUM : self.pi_input_size] = action_vector[i]
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x_tensor

    def _get_current_policy(self, state: StateData, action: ActionData, is_train: bool = True) -> torch.Tensor:
        """状態s, 行動aに対する現在の方策pi(s, a)を取得する"""
        input_tensor = self._create_input_tensor(state).to(self.device)
        if is_train:
            self.pi.train()
        else:
            self.pi.eval()

        action_values = self.pi(input_tensor)
        action_prob = torch.softmax(action_values, dim=-1)
        return action_prob

    def _compute_returns_and_advantages(
        self,
        state_values: list[float],
        rewards: list[float],
        dones: list[bool],
        gamma: float = 0.9,
        lambda_gae: float = 0.95,
    ) -> tuple[list[torch.float32], list[torch.float32]]:
        td_errors = [
            rewards[i] + gamma * state_values[i + 1] - state_values[i] if not dones[i] else rewards[i] - state_values[i]
            for i in range(len(rewards))
        ]
        # done = Trueの時 → td_error = r - v
        # breakpoint()  # len(rewards)とlen(td_errors)を確認 → rewardsの方が1つ多い | donesの中身も確認

        # GAEの計算
        gaes = [None for t in range(len(td_errors))]
        for i in reversed(range(len(td_errors))):
            if not dones[i]:
                gaes[i] = td_errors[i] + gamma * lambda_gae * (1 - dones[i]) * gaes[i + 1]
            else:
                gaes[i] = td_errors[i]

        returns = [
            torch.tensor(gae + state_value, dtype=torch.float32).to(self.device)
            for gae, state_value in zip(gaes, state_values)
        ]

        # advantage
        mean_gae = torch.mean(torch.stack(gaes))
        std_gae = torch.std(torch.stack(gaes)) + 1e-8
        advantages = [gae - mean_gae / std_gae for gae in gaes]

        # breakpoint()  # gaeの中身を確認 | returnsの長さを確認
        return returns, advantages

    def _calc_entropy(self, action_prob: list[torch.Tensor]) -> torch.Tensor:
        """エントロピーを計算する"""
        # dist = Categorical(action_prob)
        sum_entropy = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        for p in action_prob:
            sum_entropy = sum_entropy + p * torch.log(p + 1e-8)
        return sum_entropy / len(action_prob)
