import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import argparse
from pprint import pprint

from ygo.udi_io import UdiIO
from ygo.util.text import TextUtil
from ygo.models import DuelEndData
from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest, CommandEntry
from ygo import constants

from src.reward_functions.base_reward_func import BaseRewardFunction
from src.reward_functions.normal_reward_func import NormalRewardFunction
from src.ygo_env_wrapper.action_data import ActionData


class YgoEnv:
    """udiから取得するygo環境のラッパー"""
    def __init__(self, udi_io: UdiIO):
        self.udi_io = udi_io  # MDクライアントの情報参照

        self.reward_func = NormalRewardFunction(udi_io, is_normalized=True) # 勝ち:1.0、負け:-1.0、そのほか:0.0を返す報酬関数
    
    def reset(self):
        return self.step(None)

    def step(
        self, action_data: ActionData, 
    ) -> dict:
        """
        コマンド(cmd_index)を実行し、次状態を返す
        - デュエルスタートかどうか： "is_duel_start"
        - デュエルエンドかどうか: "is_duel_end"
        - アクション選択が必要かどうか: "is_cmd_required"
        - デュエル終了時の結果データ: "duel_end_result"
        - 状態: "state"
        - コマンドリクエスト: "command_request"
        - 報酬: "reward"
        """
        # コマンドの実行
        reward: float = 0.0
        if action_data is not None:
            # print("send cmd")
            cmd_index = action_data.command_index
            # cmd_indexの実行
            self.udi_io.output_command(cmd_index)
            # 報酬を計算
            reward = self.reward_func.eval(action_data=action_data)
    
        try:
            if (not self.udi_io.input()):
                pass
            elif (self.udi_io.duel_data != {}):
                state: DuelStateData = self.udi_io.get_duel_state_data()  # 次状態
                command_request: CommandRequest = self.udi_io.get_command_request()  # 選択可能なコマンドリストなどの情報
                

                
        except Exception as e:
            print(e)
            sys.exit()
        
    
        # set each flags
        is_duel_start: bool = self.udi_io.is_duel_start()
        is_duel_end: bool = self.udi_io.is_duel_end()
        is_cmd_required: bool = self.udi_io.is_command_required()
        duel_end_data: DuelEndData = self.udi_io.get_duel_end_data() if is_duel_end else None
        
        result_dict = {
            "is_duel_start": is_duel_start,
            "is_duel_end": is_duel_end,
            "is_cmd_required": is_cmd_required,
            "duel_end_data": duel_end_data,
            "state": state,
            "command_request": command_request,
            "reward": reward,
        }
        # print(f"is_cmd_required: {self.udi_io.is_command_required()}, current_phase: {state.general_data.current_phase}")
        
        return result_dict


def cli_player(result:dict) -> ActionData:
    """CLIからコマンドインデックスう入力できるテスト用のプレイヤー"""
    # コマンドCLI入力
    can_send = False
    cmd_index = -1

    while not can_send:
        print(f"select your action: [0 - {len(result["command_request"].commands)-1}]")
        for i, cmd_entry in enumerate(result["command_request"].commands):
            print(f"command {i}: {text_util.get_command_entry_text(cmd_entry)}")

        try:
            cmd_index = int(input())
            if 0 <= cmd_index < len(result["command_request"].commands):
                can_send = True

        except Exception as e:
            can_send = False
            print(f"Invalid command index. Select cmd_index in 0-{len(result["command_request"].commands)-1}")
    
    # コマンド生成
    state = result["state"]
    cmd_request = result["command_request"]
    cmd_entry = cmd_request.commands[cmd_index]
    action_data = ActionData(state=state, command_request=cmd_request, command_entry=cmd_entry)
    return action_data

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
    text_util = TextUtil()

    env = YgoEnv(udi_io=udi_io)
    print("finish initialized")

    action_data = None
    result = env.reset() # 初期化

    while True:
        # デュエルの開始 or 終了
        if result["is_duel_start"]:
            print("Duel Start")
        elif result["is_duel_end"]:
            print("Duel End")
            print(result["duel_end_data"])
        
        action_data = None
        if result["is_cmd_required"]:
            
            # 行動選択
            action_data = cli_player(result=result)
        
        result = env.step(action_data=action_data)