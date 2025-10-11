import argparse
import numpy as np

import wandb

from src.env.state_data import StateData
from src.env.ygo_env import YgoEnv

from src.agents.random_agent.random_agent import RandomAgent
from src.agents.dqn_agent.dqn_agent import DQNAgent

# Instance-1でのDuelSimulatorの起動コマンド
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp_host", type=str, default="10.95.102.79")
    parser.add_argument("--tcp_port", type=int, default=52100)
    parser.add_argument("--connect", choices=["Socket", "gRPC"], default="gRPC")
    args = parser.parse_args()

    agent = RandomAgent()

    env = YgoEnv({"tcp_host": args.tcp_host, "tcp_port": args.tcp_port, "gRPC": args.connect == "gRPC"})

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
