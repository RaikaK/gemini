import argparse
from datetime import datetime

import wandb

import src.config as config
from src.env.state_data import StateData
from src.env.ygo_env import YgoEnv

from src.agents.noah.agent import Noah


def _handle_episode_end(
    env: YgoEnv,
    next_state: StateData,
    stats: dict,
) -> StateData:
    """
    エピソード終了処理を行う。
    Args:
        env (YgoEnv): 環境
        next_state (StateData): 次の状態データ
        stats (dict): 統計情報
    """
    # 集計
    stats["episode"] += 1
    is_win: bool = next_state.reward == 1.0

    if is_win:
        stats["win_count"] += 1

    # 勝率計算
    win_rate: float = stats["win_count"] / stats["episode"]

    # ログ出力
    print(f"Episode {stats['episode']}")
    print(f"  Result            : {'Win' if is_win else 'Lose'}")
    print(f"  Win Rate (Total)  : {win_rate:.1%} ({stats['win_count']}/{stats['episode']})")

    # ログ記録
    wandb.log(
        {
            "episode": stats["episode"],
            "win_rate": win_rate,
            "is_win": int(is_win),
        }
    )

    # ログ保存
    env.udi_io.flush_udi_logs()

    return env.reset()


def main() -> None:
    """
    メイン関数
    """
    # コマンドライン引数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_demo", action="store_true")
    parser.add_argument("--tcp_port", type=int, default=60000)
    parser.add_argument("--group", type=str, default="noah_vs_cpu")
    parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args: argparse.Namespace = parser.parse_args()

    # WandB初期化
    wandb.init(
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
        group=args.group,
        name=args.name,
    )

    # エージェント初期化
    agent: Noah = Noah(save_demo=args.save_demo)

    # 環境初期化
    env: YgoEnv = YgoEnv(tcp_host="10.95.102.79", tcp_port=args.tcp_port, use_grpc=True, use_gui=False)

    if args.save_demo:
        env.udi_io.start_udi_logging(log_dir=config.RULE_UDI_LOG_DIR)

    state: StateData = env.reset()

    # 統計管理初期化
    stats: dict = {
        "episode": 0,
        "win_count": 0,
    }

    # デュエルループ
    while True:
        # 行動選択
        action, _ = agent.select_action(state=state)

        # 環境更新
        next_state: StateData = env.step(action=action)

        # エージェント更新
        agent.update(state=state, action=action, next_state=next_state, info=None)

        # デュエル終了時
        if next_state.is_duel_end:
            state = _handle_episode_end(
                env=env,
                next_state=next_state,
                stats=stats,
            )

        else:
            state = next_state


if __name__ == "__main__":
    main()
