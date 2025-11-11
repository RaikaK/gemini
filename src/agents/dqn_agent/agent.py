import os
import random
import numpy as np
import torch
import copy
import datetime
from typing import cast

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData

from ygo.constants import FinishType
from ygo.models.command_request import CommandEntry, CommandRequest
from src.agents.dqn_agent.sample_tensors import (
    set_action_vector,
    set_board_vector,
    BOARD_NUM,
    ACTION_NUM,
    INFO_NUM,
)
from src.common.sample_mlp_model import Dnn
from src.agents.supervised_laerned_agent.model_loader import load_torch_model
from src.agents.dqn_agent.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(
        self,
        gamma=0.9,
        lr=1e-6,
        epsilon=0.1,
        buffer_size=int(1e5),
        batch_size: int = 128,
        sync_interval: int = 100,
        epochs_on_update: int = 16,
        model_save_dir="params",
        model_file_name="simple_dqn_{now}.pth",
        save_model_interval: int = 64,
        init_model_params_path: str | None = None,
    ):
        print("DQNAgent")
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        # Output Only Q-value based on State&Action(CommandEntry)
        # (duel_state_data + command_entry) -> DNN -> Q-value
        self.input_size = BOARD_NUM + INFO_NUM + ACTION_NUM
        self.output_size = 1
        self.dqn: torch.nn.Module = (
            load_torch_model(model_path=init_model_params_path)
            if init_model_params_path is not None
            else Dnn(input_size=self.input_size, output_size=self.output_size)
        )
        self.target_net: torch.nn.Module = (
            load_torch_model(model_path=init_model_params_path)
            if init_model_params_path is not None
            else Dnn(input_size=self.input_size, output_size=self.output_size)
        )

        # モデルの保存に関する設定
        self.model_file_name = model_file_name
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_save_dir = os.path.join(current_dir, model_save_dir)
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.save_model_interval = save_model_interval
        self.cnt_save_model_interval = 0

        # 学習に関する設定
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.dqn.parameters(), lr=self.lr)

        # 経験再生
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        # 1episode分のデータを管理して毎回クリアする
        self.replay_short_memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        # target_netの更新頻度 (predict()をsync_interval回呼び出した後、更新する)
        self.sync_interval = sync_interval
        self.num_update = 0

        # 1回のUpdateで学習する回数
        self.epochs_on_update = epochs_on_update

        # GPU設定
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"device: {self.device}")
        self.dqn.to(self.device)
        self.target_net.to(self.device)
        return

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        # epsilon-greedy方による行動選択
        is_explore = np.random.rand()
        if is_explore <= self.epsilon:
            cmd_request: CommandRequest = state.command_request
            selected_cmd_entry: CommandEntry = random.choice(cmd_request.commands)
            action = ActionData(command_request=cmd_request, command_entry=selected_cmd_entry)
            return action, None
        else:
            return self._predict(state=state), None

    def update(
        self,
        state: StateData,
        action: ActionData,
        next_state: StateData,
        info: dict | None,
    ) -> dict | None:
        if action is None:
            return None

        reward = next_state.reward
        done = next_state.is_duel_end

        self.replay_short_memory.add(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            info=info,
        )

        if not done:
            return None
        # episode終了時に学習を行う
        self.replay_buffer.extend(
            self.replay_short_memory.buffer
        )  # おおもとの経験再生に1Episode分の経験データをすべて追加
        self.replay_short_memory.clear()  # 小メモリはクリアする

        batch_data = self.replay_buffer.get_batch()
        if batch_data is None:
            return None

        self.dqn.train()  # nn.Moduleの学習モード設定
        losses = []
        for _ in range(self.epochs_on_update):
            # dqnに入力するinpute_tensorバッチを作成する
            input_batch = []
            rewards = []
            dones = []
            next_states = []
            for replay_data in batch_data:
                # Replayデータから情報を抜き取る
                st: StateData = replay_data["state"]
                act: ActionData = replay_data["action"]
                rwd: float = replay_data["reward"]
                terminated: bool = replay_data["done"]
                next_st: StateData = replay_data["next_state"]

                # 状態のテンソルとアクションのテンソルを用意して、入力テンソルを生成
                input_ret = self._convert_state_like_sample_ai(st)
                board_vector = set_board_vector(input_ret)
                action_vector = set_action_vector(input_ret)[act.command_index]

                x = np.empty((self.input_size), dtype=np.float32)
                x[0 : BOARD_NUM + INFO_NUM] = board_vector
                x[BOARD_NUM + INFO_NUM : self.input_size] = action_vector

                input_batch.append(torch.tensor(x, dtype=torch.float32))

                # rewardやdone, next_stateなども保持
                rewards.append(rwd)
                dones.append(terminated)
                next_states.append(next_st)
            input_batch_tensor = torch.stack(input_batch).to(self.device)

            targets = self._calc_targets(next_states=next_states, rewards=rewards, dones=dones)
            targets = targets.to(self.device).unsqueeze(1)

            # 損失を計算
            loss = self.loss_func(self.dqn(input_batch_tensor), targets)

            # パラメータの更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.mean())

        loss = np.average(np.array([l.detach().cpu().numpy() for l in losses])) if len(losses) > 0 else 0
        print(f"mean loss: {loss}")

        # target_netを更新
        self._sync_qnet()
        self._save_model_params()

        finish_type = cast(FinishType, next_state.duel_end_data.finish_type) if next_state.duel_end_data else None
        log_dict = {"loss": loss, "reward": reward, "finish_type": finish_type}
        return log_dict

    def _predict(self, state: StateData) -> ActionData:
        """
        stateとCommandEntryのテンソルをconcatし、Q値を出力し、最大のQ値であるCommandEntryとなるActionDataを返す
        self.dqnからの出力
        """
        command_request: CommandRequest = state.command_request
        # DuelStateData Tensor
        input_ret = self._convert_state_like_sample_ai(state)
        board_vector = set_board_vector(input_ret)
        action_vector = set_action_vector(input_ret)
        cmd_count = len(state.command_request.commands)
        x = np.empty((cmd_count, self.input_size), dtype=np.float32)
        for i in range(cmd_count):
            x[i][0 : BOARD_NUM + INFO_NUM] = board_vector
            x[i][BOARD_NUM + INFO_NUM : self.input_size] = action_vector[i]

        input_batch_tensor = torch.tensor(x).to(self.device)
        self.dqn.eval()
        qs: torch.Tensor = self.dqn(input_batch_tensor)

        max_qs_index = np.argmax(qs.cpu().detach().numpy())

        action = ActionData(
            command_request=command_request,
            command_entry=command_request.commands[max_qs_index],
        )
        return action

    def _calc_targets(self, next_states: list[StateData], rewards: list[float], dones: list[bool]) -> torch.Tensor:
        """
        1. 各次状態における選択可能なコマンドの数を保持する辞書を作成
        - next_state_cmd_count = {index_next_state: num_command_entries}
        2. 各cmd_entryとnext_stateの組み合わせをもとに、input_batch_tensorを作成
        3. input_batch_tensorをまとめて、self.target_netに入力し、行動価値を出力
        4. 1で作成した辞書をもとに、
        {"next_state_index": max_q_value}の辞書を作成
        5. 各次状態におけるターゲットを計算し、torch.Tensorとして返す
        """
        # 1.
        next_state_cmd_count = {i: len(next_state.command_request.commands) for i, next_state in enumerate(next_states)}

        # 2.
        all_cmd_count = sum(list(next_state_cmd_count.values()))
        x = np.empty((all_cmd_count, self.input_size), dtype=np.float32)
        counter = 0
        for i, next_state in enumerate(next_states):
            input_ret = self._convert_state_like_sample_ai(next_state)
            board_vector = set_board_vector(input_ret)
            action_vector = set_action_vector(input_ret)
            cmd_count = next_state_cmd_count[i]
            for j in range(cmd_count):
                x[counter][0 : BOARD_NUM + INFO_NUM] = board_vector
                x[counter][BOARD_NUM + INFO_NUM : self.input_size] = action_vector[j]
                counter += 1
        input_batch_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # breakpoint()

        # 3.
        with torch.no_grad():
            next_qs = self.target_net(input_batch_tensor)

        # 4.
        next_state_max_q = {}  # 各次状態における最大のQ値を保持する辞書
        counts = list(next_state_cmd_count.values())
        start_index = 0
        for i, c in enumerate(counts):
            if dones[i]:
                next_state_max_q[i] = 0
            else:
                end_index = start_index + c
                next_state_max_q[i] = next_qs[start_index:end_index].max().item()
                start_index = end_index

        # 5.
        targets = []
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            if done:
                target = torch.tensor(reward, dtype=torch.float32)
            else:
                target = torch.tensor(reward + self.gamma * next_state_max_q[i], dtype=torch.float32)
            targets.append(target)
        return torch.stack(targets)

    def _sync_qnet(self):
        self.num_update += 1
        if self.num_update % self.sync_interval == 0:
            self.target_net = copy.deepcopy(self.dqn)
            self.num_update = 0

    def _save_model_params(self):
        self.cnt_save_model_interval += 1
        if self.cnt_save_model_interval % self.save_model_interval == 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_name = self.model_file_name.format(now=now)
            torch.save(self.dqn.state_dict(), os.path.join(self.model_save_dir, save_name))

    def _convert_state_like_sample_ai(self, state: StateData) -> list:
        input_ret = [
            [
                state.is_duel_start,
                state.is_duel_end,
                state.is_cmd_required,
                state.duel_end_data,
            ],
            state.duel_state_data.general_data,
            state.duel_state_data.duel_card_table,
            state.duel_state_data.chain_stack,
            state.command_request,
            state.command_request.commands,
        ]
        return input_ret
