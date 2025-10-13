import torch
from torch.distributions import Categorical
import numpy as np
import copy
from 

from src.common.sample_tensor import (
    BOARD_NUM,
    INFO_NUM,
    DNN_INPUT_NUM,
    set_board_vector,
    set_action_vector,
    create_input_data,
)
from src.agents.base_agent import BaseAgent
from src.env.state_data import StateData
from src.env.action_data import ActionData
from src.agents.ppo.actor_critic_model import Actor, Critic
from src.agents.ppo.rollout_buffer import RolloutBuffer



class PPOAgent(BaseAgent):
    def __init__(
        self,
        epochs_on_update: int = 32,  # 経験データを学習する回数
        clip_epsilon: float = 0.2,  # クリッピングの範囲
        gamma: float = 0.9,  # 割引率
        lambda_gae: float = 0.95,  # GAEのλパラメータ
        batch_size: int = 32,  # 各epochでの学習時のバッチサイズ
        lr: float = 1e-5,  # 学習率
        c_mse: float = 0.5,  # 状態価値損失の係数
        c_entropy: float = 0.01,  # エントロピーボーナスの係数
    ):
        super().__init__()
        self.epochs_on_update = epochs_on_update
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.batch_size = batch_size
        self.c_mse = c_mse
        self.c_entropy = c_entropy

        self.rollout_buffer = RolloutBuffer()  # 1episodeごとにクリアする

        # nn
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.lr = lr
        self.actor_input_size = DNN_INPUT_NUM
        self.critic_input_size = (
            BOARD_NUM + INFO_NUM
        )  # 状態価値はアクション情報を含まない
        self.actor = Actor(input_size=self.actor_input_size)
        self.critic = Critic(input_size=self.critic_input_size)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.mse_loss = torch.nn.MSELoss()

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        # 入力テンソル
        x_tensor = self._create_input_tensor(state).to(self.device)
        # 推論モードにして、アクションの確率分布を生成
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(x_tensor)
            probs = torch.softmax(action_values, dim=-1).flatten()
            probs /= torch.sum(probs)
            # breakpoint()  # shapeとcmd_countの確認を行う # どこでunsqeezeを実行するか見極める
            action_index = np.random.choice(len(probs), p=probs.detach().cpu().numpy())
            action_prob: np.ndarray = probs[action_index].detach().cpu().numpy() + 1e-10

        # 状態価値を一緒に出力
        self.critic.eval()
        with torch.no_grad():
            board_vector = set_board_vector(create_input_data(state))
            board_vector_tensor = torch.tensor(board_vector, dtype=torch.float32).to(
                self.device
            )
            state_value: np.ndarray = (
                self.critic(board_vector_tensor).cpu().detach().numpy()
            )

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
        rollout_data = self.rollout_buffer.rollout()
        # self.rollout_buffer.clear()
        rollout_states: list[StateData] = rollout_data["states"]
        rollout_actions: list[ActionData] = rollout_data["actions"]
        rollout_rewards = rollout_data["rewards"]
        rollout_dones = rollout_data["dones"]
        # next_states = data["next_states"]
        rollout_action_probs = [d["action_prob"] for d in rollout_data["infos"]]
        rollout_state_values = [d["state_value"] for d in rollout_data["infos"]]
        # returns, advantages = self._compute_returns_and_advantages(
        #     state_values=state_values,
        #     rewards=rewards,
        #     dones=dones,
        #     gamma=self.gamma,
        #     lambda_gae=self.lambda_gae,
        # )

        indices = np.arange(len(states)) # rolloutされたデータ数
        np.random.shuffle(indices)

        losses_actor = []
        losses_critic = []
        for _ in range(self.epochs_on_update):
            for start in range(0, len(states), self.batch_size):
                # バッチ学習のためのランダムなデータ抽出
                idxes = indices[start: start+self.batch_size]
                states: list[StateData] = rollout_states[idxes]
                actions: list[ActionData] = rollout_actions[idxes]
                rewards: list[float] = rollout_rewards[idxes]
                dones: list[bool] = rollout_dones[idxes]
                action_probs: np.ndarray = np.array(rollout_action_probs[idxes])
                state_values: list[float] = rollout_state_values[idxes]
                returns, advantages = self._compute_returns_and_advantages(
                    state_values=state_values,
                    rewards=rewards,
                    dones=dones,
                    gamma=self.gamma,
                    lambda_gae=self.lambda_gae,
                )
                breakpoint() # idxesの中身を確認
                # データ収集時の方策
                log_pi_old = torch.log(torch.tensor(action_probs).to(self.device))
                log_pi = torch.log(
                    torch.stack(
                        [
                            self._get_current_policy(s)[a.command_index]
                            for s, a in zip(states, actions)
                        ]
                    )
                ).to(self.device)
                # 方策比をクリッピング
                ratio_unclipped = torch.exp(log_pi - log_pi_old)
                ratio_clip = torch.clamp(
                    ratio_unclipped, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )

                # エントロピーボーナス
                dist_entropy = torch.stack(
                    [self._calc_entropy(state=state) for state in states]
                ).to(self.device)

                # breakpoint(s)
                loss_actor = (
                    -torch.min(ratio_unclipped * advantages, ratio_clip * advantages)
                    - self.c_entropy * dist_entropy
                )
                loss_actor = loss_actor.mean()
                self.optim_actor.zero_grad()
                loss_actor.backward()
                self.optim_actor.step()

                # Criticの損失
                board_tensor = torch.tensor(
                    np.array(
                        [set_board_vector(create_input_data(state)) for state in states]
                    )
                ).to(self.device)
                # breakpoint()  # board_tensorのshapeとreturnsのshapeを確認
                loss_critic = self.mse_loss(self.critic(board_tensor), returns)
                loss_critic = loss_critic.mean()
                self.optim_critic.zero_grad()
                loss_critic.backward()
                self.optim_critic.step()

                # lossesに追加
                losses_actor.append(loss_actor)
                losses_critic.append(loss_critic)

        duel_end_data = next_state.duel_end_data
        finish_type = duel_end_data.finish_type if duel_end_data is not None else 0
        return {
            "loss_actor": np.average(losses_actor) if len(losses_actor) > 0 else 0,
            "loss_critic": np.average(losses_critic) if len(losses_critic) > 0 else 0,
            "reward": reward,
            "ave_reward": np.average(rewards) if len(rewards) > 0 else 0,
            "finish_type(1:Normal|2:NoDeck)": finish_type,
        }

    def _create_input_tensor(self, state: StateData) -> torch.Tensor:
        """状態s, 行動aに対する現在の方策pi(s, a)を取得する"""
        input_data = create_input_data(state)
        board_vector = set_board_vector(input_data)
        action_vector = set_action_vector(input_data)
        command_request = state.command_request
        cmd_count = len(command_request.commands)
        # テンソルの作成
        x = np.empty((cmd_count, self.actor_input_size), dtype=np.float32)
        for i in range(cmd_count):
            x[i][0 : BOARD_NUM + INFO_NUM] = board_vector
            x[i][BOARD_NUM + INFO_NUM : self.actor_input_size] = action_vector[i]
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x_tensor

    def _get_current_policy(
        self, state: StateData, is_train: bool = True
    ) -> torch.Tensor:
        """状態sの時、各行動の確率分布を返す"""
        input_tensor = self._create_input_tensor(state)
        if is_train:
            self.actor.train()
        else:
            self.actor.eval()

        action_values = self.actor(input_tensor)
        action_probs = torch.softmax(action_values, dim=-1)
        return action_probs / torch.sum(action_probs)

    def _compute_returns_and_advantages(
        self,
        state_values: list[float],
        rewards: list[float],
        dones: list[bool],
        gamma: float = 0.9,
        lambda_gae: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        - V_target(=returns)とAdvantageを計算する
        - detach()済み
        """
        # td_errors = [
        #     rewards[i] + gamma * state_values[i + 1] - state_values[i]
        #     if not dones[i]
        #     else rewards[i] - state_values[i]
        #     for i in range(len(rewards))
        # ]
        td_errors = [
            rewards[i]
            + gamma * state_values[i + 1] * (1 - int(dones[i]))
            - state_values[i]
            for i in range(len(rewards))
        ]
        # done = Trueの時 → td_error = r - v
        # breakpoint()  # len(rewards)とlen(td_errors)を確認 → rewardsの方が1つ多い | donesの中身も確認

        # GAEの計算
        gaes = [None for t in range(len(td_errors))]
        for i in reversed(range(len(td_errors))):
            if not dones[i]:
                gaes[i] = (
                    td_errors[i] + gamma * lambda_gae * (1 - dones[i]) * gaes[i + 1]
                )
            else:
                gaes[i] = td_errors[i]
        # 期待収益の計算
        returns = [gae + state_value for gae, state_value in zip(gaes, state_values)]

        # 平均と標準偏差で正規化してAdvantageとする
        mean_gae = np.mean(gaes)
        std_gae = np.std(gaes) + 1e-8
        advantages = [(gae - mean_gae) / std_gae for gae in gaes]

        return torch.tensor(np.array(returns)).detach().to(self.device), torch.tensor(
            np.array(advantages)
        ).detach().to(self.device)

    def _calc_entropy(self, state: StateData) -> torch.Tensor:
        """状態s、行動aの時の方策から、エントロピーを計算する"""
        policy = self._get_current_policy(state)
        return -torch.sum(policy * torch.log(policy))
