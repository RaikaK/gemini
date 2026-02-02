import atexit
import signal
import subprocess
import threading

from src.env.state_data import StateData
from src.env.ygo_env import YgoEnv

from src.agents.noah.agent import Noah


# 対戦用AIプロセス管理
_opponent_processes: list[subprocess.Popen] = []


def _run_opponent_ai(command: list[str]) -> None:
    """
    対戦用AIのコマンドを実行する。

    Args:
        command (list[str]): 実行コマンド
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    _opponent_processes.append(process)
    process.wait()


def _start_opponent_ais(tcp_host: str) -> None:
    """
    対戦用AIを起動する。

    Args:
        tcp_host (str): TCPホスト名
    """
    commands: list[list[str]] = [
        ["C:\\UDI\\samples\\etc\\opponent_ai_cython\\Step2_run_ModelCy.bat"],
        [
            "C:\\UDI\\samples\\etc\\opponent_ai_cython\\Step3_run_ClientCy.bat",
            tcp_host,
        ],
    ]

    for command in commands:
        thread = threading.Thread(
            target=_run_opponent_ai,
            args=(command,),
            daemon=True,
        )
        thread.start()


def _stop_opponent_ais() -> None:
    """
    対戦用AIを停止する。
    """
    for process in _opponent_processes:
        if process.poll() is None:
            try:
                process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))

            except Exception:
                pass


atexit.register(_stop_opponent_ais)


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

    return env.reset()


def main() -> None:
    """
    メイン関数
    """
    # エージェント初期化
    agent: Noah = Noah(save_demo=False)

    # 環境初期化
    env: YgoEnv = YgoEnv(tcp_host="10.95.102.79", tcp_port=50001, use_grpc=False, use_gui=False)

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
    _start_opponent_ais(tcp_host="10.95.102.79")
    main()
