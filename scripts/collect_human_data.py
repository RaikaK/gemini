import argparse

from src.agents.human.agent import HumanAgent
from src.env.ygo_env import YgoEnv

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

    state = env.reset()

    while True:
        action, info = agent.select_action(state)
        next_state = env.step(action)
        agent.update(state=state, action=action, next_state=next_state, info=info)
        state = next_state

        if state.is_duel_end:
            state = env.reset()
