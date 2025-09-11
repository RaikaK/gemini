import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import random
import numpy as np
import torch
import copy

from src.agents.base_ygo_agent import BaseYgoAgent
from src.ygo_env_wrapper.action_data import ActionData

from ygo.models.duel_state_data import DuelStateData
from ygo.models import CommandEntry, CommandRequest
from ygo.constants.enums import SelectionType


# シミュレータ起動コマンド
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1

from src.agents.dqn_agent.deep_q_network import DeepQNetwork
from src.agents.dqn_agent.replay_buffer import ReplayBuffer
from src.agents.dqn_agent.simple_tensors.duel_state_data_tensor import (
    simple_duel_state_data_tensor,
    DIM_DUEL_STATE_DATA
)
from src.agents.dqn_agent.simple_tensors.command_entry_tensor import (
    simple_command_entry_tenosr,
    DIM_COMMAND_ENTRY
)


class DQNAgent(BaseYgoAgent):
    def __init__(
        self, gamma=0.9, lr=1e-5, epsilon=0.2, buffer_size=int(1e4), batch_size: int=64, sync_interval:int=50
    ):
        print("DQNAgent")
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        # Output Only Q-value based on State&Action(CommandEntry)
        # (duel_state_data + command_entry) -> DNN -> Q-value
        input_size = DIM_DUEL_STATE_DATA+DIM_COMMAND_ENTRY
        output_size = 1 
        self.dqn = DeepQNetwork(input_size=input_size, output_size=output_size)
        self.target_net = DeepQNetwork(input_size=input_size, output_size=output_size)

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.dqn.parameters(), lr=self.lr)
        
        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size, batch_size=batch_size
        )

        # target_netの更新頻度 (predict()をsync_interval回呼び出した後、更新する)
        self.sync_interval = sync_interval
        self.num_predict = 0

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
                state=state["state"],
                command_request=cmd_request,
                command_entry=selected_cmd_entry,
            )
            return action_data
        else:
            return self.predict(state=state)
            
    def update(self, state: dict, action_data: ActionData, next_state: dict):
        if action_data is None:
            return
        reward = next_state["reward"]
        done = next_state["is_duel_end"]
        # 経験再生バッファーにデータを追加
        self.replay_buffer.add(state=state, action_data=action_data, reward=reward, done=done, next_state=next_state)
        batch_data = self.replay_buffer.get_batch()
        if batch_data is None:
            return
        
        # dqnに入力するinpute_tensorバッチを作成する
        input_batch = []
        rewards = []
        dones = []
        next_states = []
        for replay_data in batch_data:
            # Replayデータから情報を抜き取る
            state:dict = replay_data["state"]
            action_data: ActionData = replay_data["action_data"]
            reward: float = replay_data["reward"]
            done: bool = replay_data["done"]
            next_state:dict = replay_data["next_state"]
            
            # 状態のテンソルとアクションのテンソルを用意して、入力テンソルを生成
            state_tensor = simple_duel_state_data_tensor(duel_state_data=state["state"])
            cmd_entry_tensor = simple_command_entry_tenosr(command_entry=action_data.command_entry)
            input_tensor = torch.cat([state_tensor, cmd_entry_tensor])
            input_batch.append(input_tensor)

            # rewardやdone, next_stateなども保持
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
        input_batch_tensor = torch.stack(input_batch).to(self.device)
        
        # targetを計算して、GPUに送信
        targets = [
            self.calc_target(next_state, reward, done) 
            for next_state, reward, done in zip(next_states, rewards, dones) 
            # if len(next_state["command_request"].commands) > 0
        ] # 勾配情報なし
        
        targets = torch.stack(targets).to(self.device).unsqueeze(1)

        # 損失を計算
        loss = self.loss_func(self.dqn(input_batch_tensor), targets)

        # パラメータの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target_netを更新
        self.sync_qnet()

        print(f"loss: {loss.mean()}")


    def predict(self, state: dict) -> ActionData:
        """
        stateとCommandEntryのテンソルをconcatし、Q値を出力し、最大のQ値であるCommandEntryとなるActionDataを返す
        self.dqnからの出力
        """
        # print("dqn agent predict ...")
        self.num_predict += 1
        # DuelStateData Tensor
        duel_state_data: DuelStateData = state["state"]
        duel_state_data_tensor = simple_duel_state_data_tensor(duel_state_data=duel_state_data)

        command_request: CommandRequest = state["command_request"]

        input_batch = []
        for command_entry in command_request.commands:
            cmd_entry_tensor = simple_command_entry_tenosr(command_entry=command_entry)
            input_tensor = torch.cat([duel_state_data_tensor, cmd_entry_tensor], dim=0)
            input_batch.append(input_tensor)
        # breakpoint()
        input_batch_tensor = torch.stack(input_batch).to(self.device)

        qs: torch.Tensor = self.dqn(input_batch_tensor)

        max_qs_index = np.argmax(qs.cpu().detach().numpy())

        action_data = ActionData(state=duel_state_data, command_request=command_request, command_entry=command_request.commands[max_qs_index])
        return action_data
    
    def calc_target(self, next_state: dict, reward: float, done) -> torch.Tensor|None:
        """
        DQNでのターゲットとなる目的値(収益)を計算
        * next_state["command_request"].commandsは1つ以上のコマンドを有する
        """
        if done:
            return reward
        # 次状態における最適アクションを推論する
        next_q = self.calc_next_q_from_target_net(next_state=next_state)

        # 収益を算出(もしdone=Trueならばh、Rewardのみを返す)
        ret = reward + self.gamma * next_q * (1-done)
        return torch.tensor(ret, dtype=torch.float32)

    def sync_qnet(self):
        if self.num_predict % self.sync_interval == 0:
            self.target_net = copy.deepcopy(self.dqn)
            self.num_predict = 0

    def calc_next_q_from_target_net(self, next_state:dict) -> np.float32:
        """次状態において、選択可能な行動ActionDataのうち、最大の値であるQ値を返す"""
        # 次状態のDuelStateDataをテンソルに変換
        next_duel_state_data: DuelStateData = next_state["state"]
        next_duel_state_tensor = simple_duel_state_data_tensor(duel_state_data=next_duel_state_data)

        # CommandRequestに含まれる選択可能なCmdEntryのうち、最大のQ値を計算
        next_cmd_request: CommandRequest = next_state["command_request"]

        input_batch = [] # Batch処理するための空のリスト
        for cmd_entry in next_cmd_request.commands:
            # 各CmdEntryをテンソル化し、入力テンソルを作成
            cmd_entry_tensor = simple_command_entry_tenosr(command_entry=cmd_entry)
            input_tensor = torch.cat([next_duel_state_tensor, cmd_entry_tensor])
            input_batch.append(input_tensor)
        # breakpoint()
        input_batch_tensor = torch.stack(input_batch).to(self.device)
        with torch.no_grad():
            next_qs = self.target_net(input_batch_tensor)
        
        return np.max(next_qs.cpu().detach().numpy())