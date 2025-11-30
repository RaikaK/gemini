import argparse
from datetime import datetime

import wandb

from src.agents.hierarchical.agent import HierarchicalAgent
import src.config as config
from src.env.state_data import StateData
from src.env.ygo_env import YgoEnv


def main() -> None:
    """
    メイン関数
    """
    # コマンドライン引数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp_host", type=str, default="10.95.102.79")
    parser.add_argument("--tcp_port", type=int, default=52100)
    parser.add_argument("--connect", choices=["Socket", "gRPC"], default="gRPC")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--group", type=str, default="vs_CPU")
    parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    # エージェント初期化
    model_path = config.MODELS_DIR / "hierarchical-agent_sl" / "20251201_000000" / "epoch_0001.pth"
    agent = HierarchicalAgent(model_path=model_path)

    # 環境初期化
    env = YgoEnv(tcp_host=args.tcp_host, tcp_port=args.tcp_port, use_grpc=args.connect == "gRPC", use_gui=args.use_gui)
    state: StateData = env.reset()

    # wandb初期化
    if not args.no_wandb:
        wandb.init(
            entity=config.WANDB_ENTITY,
            project=config.WANDB_PROJECT,
            group=args.group,
            name=args.name,
            config={"model_path": str(model_path)},
        )

    # デュエルループ
    episode = 0
    win_count = 0

    while True:
        # 行動選択
        action, info = agent.select_action(state)

        # 環境更新
        next_state: StateData = env.step(action)

        # エージェント更新
        log: dict | None = agent.update(state=state, action=action, next_state=next_state, info=info)

        # ログ記録
        if log is not None and not args.no_wandb:
            wandb.log(log)

        state = next_state

        # デュエル終了時
        if state.is_duel_end:
            # 集計
            episode += 1
            is_win = state.reward == 1.0

            if is_win:
                win_count += 1

            win_rate = win_count / episode

            # ログ出力
            print(
                f"Episode: {episode:<4} | Win Rate: {win_rate:.1%} ({win_count}/{episode}) | Result: {'Win' if is_win else 'Lose'}"
            )

            # ログ記録
            if not args.no_wandb:
                wandb.log({"episode": episode, "win_rate": win_rate, "is_win": is_win})

            state = env.reset()


if __name__ == "__main__":
    main()
