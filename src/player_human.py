import argparse
import numpy as np

import wandb

from src.env.state_data import StateData
from src.env.ygo_env import YgoEnv

from src.agents.random.agent import RandomAgent
from src.agents.dqn_agent.dqn_agent import DQNAgent
from src.agents.human.agent import HumanAgent

# Instance-1でのDuelSimulatorの起動コマンド
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp_host", type=str, default="10.95.102.79")
    parser.add_argument("--tcp_port", type=int, default=52100)
    parser.add_argument("--connect", choices=["Socket", "gRPC"], default="gRPC")
    parser.add_argument("--use_gui", action="store_true")
    args = parser.parse_args()

    env = YgoEnv(tcp_host=args.tcp_host, tcp_port=args.tcp_port, use_grpc=args.connect == "gRPC", use_gui=args.use_gui)

    if env.command_queue is None:
        raise RuntimeError("GUIモードで起動してください。")
    agent = HumanAgent(command_queue=env.command_queue)

    episode = 0
    state: StateData = env.reset()

    while True:
        action, info_dict = agent.select_action(state)

        next_state: StateData = env.step(action)

        log_dict: dict | None = agent.update(state=state, action=action, next_state=next_state, info=info_dict)

        state = next_state

        if state.is_duel_end:
            episode += 1
            state = env.reset()
