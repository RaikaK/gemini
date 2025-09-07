import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import argparse
import random
import time

from ygo.udi_io import UdiIO
from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest, CommandEntry
from ygo import constants

from src.reward_functions.base_reward_func import BaseRewardFunction
from src.reward_functions.normal_reward_func import NormalRewardFunction
from src.ygo_env_wrapper.action_data import ActionData

WAITING_TIME = 1

class YgoEnv:
    def __init__(self, udi_io: UdiIO, reward_func: BaseRewardFunction):
        self.udi_io = udi_io  # MDクライアントの情報参照

        self.reward_func = reward_func  # 報酬関数
        # 何も指定がない場合は、 NormalRewardFunctionを使用
        if self.reward_func is None:
            self.reward_func = NormalRewardFunction(udi_io, is_normalized=True)

        
        self.is_cmd_required = False
        # self.udi_io.wait(lambda _: self._mainloop)

    def step(
        self, action_data: ActionData, 
    ) -> dict[str, DuelStateData | CommandRequest | float | bool]:
        """
        コマンド(cmd_index)を実行し、
        is_cmd_required==Trueのときreturnする
        - 次状態: "next_state"
        - コマンドリクエスト: "command_request"
        - 報酬: "reward"
        - 終了フラグ: "done"
        を返す
        """
        
        while not self.udi_io.input():
            print("wating udi_io.input() ")
            time.sleep(WAITING_TIME)

        if action_data is not None:
            cmd_index = action_data.command_index
            # cmd_indexの実行
            self.udi_io.output_command(cmd_index)
        
        
        # ##############
        # コマンド実行後
        # ##############
        # コマンド要求まで待機
        while not self.udi_io.is_command_required():
            time.sleep(WAITING_TIME)
            print("waiting ...")
        
        
        next_state: DuelStateData = self.udi_io.get_duel_state_data()  # 次状態
        command_request = (
            self.udi_io.get_command_request()
        )  # 選択可能なコマンドリストなどの情報

        # 報酬を計算
        reward: float = self.reward_func.eval(action_data=action_data)

        done = self.udi_io.is_duel_end()  # 終了フラグ


        result_dict = {
            "next_state": next_state,
            "command_request": command_request,
            "reward": reward,
            "done": done,
        }
        
        return result_dict



if __name__ == "__main__":
    print("起動コマンド（通常）")
    print("python SampleAi.py")
    print("起動コマンド（継続）")
    print("python SampleAi.py --LoadWeightName save_20250101_000000_0000.pth")

    print("設定テスト開始")
    x = constants.PosId.EX_R_MONSTER
    print(x)
    print("起動テスト完了")
    print("引数を読み込みます。")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tcpport',        type=int, default=52010)
    parser.add_argument('--tcphost',        type=str, default="10.95.102.79")
    parser.add_argument("-g", "--gRPC",     action="store_true", help="using gRPC")
    parser.add_argument('--RandomPlayer',   type=int, default=0, help='0:AI 1:RandomPalyer')
    parser.add_argument('--LoadWeightName', type=str, default=None)
    parser.add_argument('-x',               type=int, default=0, help='Dummy')
    args = parser.parse_args()
    RandomPlayerFlag = args.RandomPlayer

    print("ＵＤＩの設定を行います。")
    connect = UdiIO.Connect.SOCKET
    if args.gRPC:
        connect = UdiIO.Connect.GRPC

    #UDIの初期化
    udi_io = UdiIO(tcpport=args.tcpport, tcp_host=args.tcphost, connect=connect, api_version=1)
    #UDIのログは大量に出るため、出力しないようにする
    udi_io.log_response_history = False

    env = YgoEnv(udi_io=udi_io, reward_func=None)
    print("finish initialized")
    
    initial_result = env.step(None)
    print("finish first step()")

    done = initial_result["done"]
    while not done:
        cmd = input(f"select your aciton: [0 - {initial_result["command_request"].commands}]")
        result = env.step(cmd)
        breakpoint()

        done = result["done"]