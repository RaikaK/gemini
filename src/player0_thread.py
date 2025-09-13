import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import argparse
import numpy as np

from ygo import constants
from ygo.udi_io import UdiIO

from src.ygo_env_wrapper.action_data import ActionData
from src.ygo_env_wrapper.ygo_env import YgoEnv
from src.agents.random_agent.random_agent import RandomAgent
from src.agents.dqn_agent.dqn_agent import DQNAgent

# Instance-1でのDuelSimulatorの起動コマンド
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1

WAITING_TIME = 1e-4

if __name__ == "__main__":
    print("起動コマンド（通常）")
    print("python player0_thread.py")
    print("起動コマンド（継続）")

    print("設定テスト開始")
    x = constants.PosId.EX_R_MONSTER
    print(x)
    print("起動テスト完了")
    print("引数を読み込みます。")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcpport", type=int, default=52010)
    parser.add_argument("--tcphost", type=str, default="10.95.102.79")
    parser.add_argument(
        "-g", "--gRPC", action="store_true", help="using gRPC", default="-g"
    )
    parser.add_argument(
        "--RandomPlayer", type=int, default=0, help="0:AI 1:RandomPalyer"
    )
    parser.add_argument("--LoadWeightName", type=str, default=None)
    parser.add_argument("-x", type=int, default=0, help="Dummy")
    args = parser.parse_args()
    print("ＵＤＩの設定を行います。")
    connect = UdiIO.Connect.SOCKET
    if args.gRPC:
        connect = UdiIO.Connect.GRPC

    # UDIの初期化
    udi_io = UdiIO(
        tcpport=args.tcpport, tcp_host=args.tcphost, connect=connect, api_version=1
    )
    # UDIのログは大量に出るため、出力しないようにする
    udi_io.log_response_history = False

    # エージェントの用意
    # agent = RandomAgent()
    agent = DQNAgent()

    # 環境ラッパーの起動
    env = YgoEnv(udi_io=udi_io)
    
    # wandb setting
    import wandb
    entity = "ygo-ai" # TeamAccount名
    project = "Random-vs-DQN"
    wandb.init(project=project, entity=entity)

    episode = 0
    reward_history = []
    state = env.reset()
    while True:
        action_data = agent.select_action(state)
        
        state = env.step(action_data)

        # エージェントの学習
        log_dict = agent.update(state=state, action_data=action_data, next_state=state)

        # log
        if log_dict is not None:
            wandb.log(log_dict)
        reward_history.append(state["reward"])

        
        if state["is_duel_end"]:
            ave_reward = np.average(reward_history) if len(reward_history) > 0 else 0
            wandb.log({"ave_reward": ave_reward})
            reward_history.clear()
            
            print(f"episode: {episode} | ave_reward: {ave_reward} | result: {state["duel_end_data"]}")
            episode += 1
            state = env.reset()
    
    wandb.finish()