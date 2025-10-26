import argparse

from src.agents.human.agent import HumanAgent
from src.env.state_data import StateData
from src.env.ygo_env import YgoEnv

# シュミレータ起動コマンド
# DuelSimulator.exe --player_type0 Human --player_type1 CPU --deck_path0 DeckData/RoyaleBE.json --deck_path1 DeckData/RoyaleBE.json --first_player 0 --lp0 8000 --lp1 8000 --hand_num0 5 --hand_num1 5 --log_level 1 --loop_num 100000 --randomize_seed true --play_reverse_duel true --exit_with_udi true --connect gRPC --grpc_deadline_seconds 60 --tcp_host0 10.95.102.79 --tcp_port0 53000 --tcp_host1 10.95.102.79 --tcp_port1 53100 --on_start_retry_connect_seconds 60

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp_host", type=str, default="10.95.102.79")
    parser.add_argument("--tcp_port", type=int, default=53000)
    parser.add_argument("--connect", choices=["Socket", "gRPC"], default="gRPC")
    args = parser.parse_args()

    env = YgoEnv(tcp_host=args.tcp_host, tcp_port=args.tcp_port, use_grpc=(args.connect == "gRPC"), use_gui=True)

    if env.command_queue is None:
        raise RuntimeError("Command queue in YgoEnv is None for HumanAgent.")

    agent = HumanAgent(command_queue=env.command_queue, save_demo=True)

    state: StateData = env.reset()

    while True:
        action, info = agent.select_action(state)
        next_state: StateData = env.step(action)
        agent.update(state=state, action=action, next_state=next_state, info=info)
        state = next_state

        if state.is_duel_end:
            state = env.reset()
