import sys
import os

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import random
import numpy as np
import torch
import copy
import datetime

from src.agents.base_ygo_agent import BaseYgoAgent
from src.ygo_env_wrapper.action_data import ActionData

from ygo.models.duel_state_data import DuelStateData
from ygo.models import CommandEntry, CommandRequest
from vendor.UDI.samples.tutorial.sample1.SampleAi import (
    SetBoardVector,
    SetActionVector,
    BoardNum,
    ActionNum,
    InforNum,
)


# シミュレータ起動コマンド
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1

from src.agents.dqn_agent.deep_q_network import DeepQNetwork
from src.agents.dqn_agent.replay_buffer import ReplayBuffer


class DQNAgent(BaseYgoAgent):
    def __init__(
        self,
        gamma=0.9,
        lr=1e-5,
        epsilon=0.2,
        buffer_size=int(1e5),
        batch_size: int = 128,
        sync_interval: int = 100,
        epochs_on_update: int = 16,
        model_save_dir="params",
        model_file_name="simple_dqn_{now}.pth",
        save_model_interval: int = 64,
    ):
        print("DQNAgent")
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        # Output Only Q-value based on State&Action(CommandEntry)
        # (duel_state_data + command_entry) -> DNN -> Q-value
        self.input_size = BoardNum + ActionNum
        self.output_size = 1
        self.dqn = DeepQNetwork(
            input_size=self.input_size, output_size=self.output_size
        )
        self.target_net = DeepQNetwork(
            input_size=self.input_size, output_size=self.output_size
        )

        # モデルの保存ン関する設定
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
        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size, batch_size=batch_size
        )

        # 1episode分のデータを管理して毎回クリアする
        self.replay_short_memory = ReplayBuffer(
            buffer_size=buffer_size, batch_size=batch_size
        )

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

    def select_action(self, state: dict) -> ActionData:
        # epsilon-greedy方による行動選択
        is_explore = np.random.rand()
        if is_explore <= self.epsilon:
            cmd_request: CommandRequest = state["command_request"]
            selected_cmd_entry: CommandEntry = random.choice(cmd_request.commands)
            action_data = ActionData(
                state=state,
                command_entry=selected_cmd_entry,
            )
            return action_data
        else:
            return self._predict(state=state)

    def update(
        self, state: dict, action_data: ActionData, next_state: dict
    ) -> dict | None:
        if action_data is None:
            return None
        reward = next_state["reward"]
        done = next_state["is_duel_end"]

        self.replay_short_memory.add(
            state=state,
            action_data=action_data,
            reward=reward,
            done=done,
            next_state=next_state,
        )

        if not done:
            return None
        # episode終了時に学習を行う
        self.replay_short_memory.update_all_reward(
            reward=reward
        )  # ゲーム終了時に報酬を更新
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
                state: dict = replay_data["state"]
                action_data: ActionData = replay_data["action_data"]
                reward: float = replay_data["reward"]
                done: bool = replay_data["done"]
                next_state: dict = replay_data["next_state"]

                # 状態のテンソルとアクションのテンソルを用意して、入力テンソルを生成
                input_ret = self._get_input_ret(state)
                board_vector = SetBoardVector(input_ret)
                action_vector = SetActionVector(input_ret)[action_data.command_index]
                x = np.empty((self.input_size), dtype=np.float32)
                x[0 : BoardNum + InforNum] = board_vector
                x[BoardNum + InforNum : self.input_size] = action_vector

                input_batch.append(torch.tensor(x, dtype=torch.float32))

                # rewardやdone, next_stateなども保持
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state)
            input_batch_tensor = torch.stack(input_batch).to(self.device)

            # targets = self.calc_targets(
            #     next_states=next_states, rewards=rewards, dones=dones
            # )
            # targets = targets.to(self.device).unsqueeze(1)
            targets = (
                torch.stack(
                    [torch.tensor(reward, dtype=torch.float32) for reward in rewards]
                )
                .to(self.device)
                .unsqueeze(1)
            )

            # 損失を計算
            loss = self.loss_func(self.dqn(input_batch_tensor), targets)

            # パラメータの更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.mean())

        loss = (
            np.average(np.array([l.detach().cpu().numpy() for l in losses]))
            if len(losses) > 0
            else 0
        )
        print(f"mean loss: {loss}")

        # target_netを更新
        self.sync_qnet()
        self.save_model_params()

        log_dict = {"loss": loss}
        return log_dict

    def _predict(self, state: dict) -> ActionData:
        """
        stateとCommandEntryのテンソルをconcatし、Q値を出力し、最大のQ値であるCommandEntryとなるActionDataを返す
        self.dqnからの出力
        """
        command_request: CommandRequest = state["command_request"]
        # DuelStateData Tensor
        input_ret = self._get_input_ret(state)
        board_vector = SetBoardVector(input_ret)
        action_vector = SetActionVector(input_ret)
        batch_num = len(state["command_request"].commands)
        x = np.empty((batch_num, self.input_size), dtype=np.float32)
        for i in range(batch_num):
            x[i][0 : BoardNum + InforNum] = board_vector
            x[i][BoardNum + InforNum : self.input_size] = action_vector[i]

        input_batch_tensor = torch.tensor(x).to(self.device)
        self.dqn.eval()
        qs: torch.Tensor = self.dqn(input_batch_tensor)

        max_qs_index = np.argmax(qs.cpu().detach().numpy())

        action_data = ActionData(
            state=state,
            command_entry=command_request.commands[max_qs_index],
        )
        return action_data

    def calc_targets(
        self, next_states: list[dict], rewards: list[float], dones: list[bool]
    ) -> torch.Tensor | None:
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
        next_state_cmd_count = {
            i: len(next_state["command_request"].commands)
            for i, next_state in enumerate(next_states)
        }

        # 2.

        input_batch = []
        for next_state in next_states:
            input_ret = self._get_input_ret(next_state)
            board_vector = SetBoardVector(input_ret)
            action_vector = SetActionVector(input_ret)
            batch_num = len(next_state["command_request"].commands)
            x = np.empty((batch_num, self.input_size), dtype=np.float32)
            input_batch.append(torch.tensor(x, dtype=torch.float32))
        input_batch_tensor = torch.stack(input_batch).to(self.device)

        # breakpoint()

        # 3.
        with torch.no_grad():
            next_qs = self.target_net(input_batch_tensor)

        # 4.
        next_state_max_q = {}  # 各次状態における最大のQ値を保持する辞書
        counts = list(next_state_cmd_count.values())
        start_index = 0
        for i, c in enumerate(counts):
            end_index = start_index + c
            next_state_max_q[i] = next_qs[start_index:end_index].max().item()
            start_index = end_index

        # 5.
        targets = []
        for i, (next_state, reward, done) in enumerate(
            zip(next_states, rewards, dones)
        ):
            if done:
                target = torch.tensor(reward)
            else:
                target = torch.tensor(reward + self.gamma * next_state_max_q[i])
            targets.append(target)
        return torch.stack(targets)

    def sync_qnet(self):
        self.num_update += 1
        if self.num_update % self.sync_interval == 0:
            self.target_net = copy.deepcopy(self.dqn)
            self.num_update = 0

    def save_model_params(self):
        self.cnt_save_model_interval += 1
        if self.cnt_save_model_interval % self.save_model_interval == 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_name = self.model_file_name.format(now=now)
            torch.save(
                self.dqn.state_dict(), os.path.join(self.model_save_dir, save_name)
            )

    def _get_input_ret(self, state: dict):
        input_ret = [
            [
                state["is_duel_start"],
                state["is_duel_end"],
                state["is_cmd_required"],
                state["duel_end_data"],
            ],
            state["state"].general_data,
            state["state"].duel_card_table,
            state["state"].chain_stack,
            state["command_request"],
            state["command_request"].commands,
        ]
        return input_ret
