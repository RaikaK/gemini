#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import argparse
import sys
import random
import datetime

import numpy as np
import torch

from ygo.udi_io import UdiIO
from ygo import constants
from ygo import models


#----------------------------------------------------------------
#テスト表示
#----------------------------------------------------------------
print("起動コマンド（通常）")
print("python SampleAi.py")
print("起動コマンド（継続）")
print("python SampleAi.py --LoadWeightName save_20250101_000000_0000.pth")

print("設定テスト開始")
x = constants.PosId.EX_R_MONSTER
print(x)
print("起動テスト完了")

#ランダムプレイヤーとの対戦例
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1
# python SampleAi.py -g --tcpport 52010 
# python SampleAi.py -g --tcpport 52011 --RandomPlayer 1

#----------------------------------------------------------------
#初期化処理等
#----------------------------------------------------------------
print("引数を読み込みます。")
parser = argparse.ArgumentParser()
parser.add_argument('--tcpport',        type=int, default=50001)
parser.add_argument('--tcphost',        type=str, default="127.0.0.1")
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

#----------------------------------------------------------------
#UDIから状態を取得
#----------------------------------------------------------------
def GetNextState():
    game_data = None
    duel_card_table = None
    chain_data = None
    command_request = None
    commands = []

    # 取得したJSONデータをファイルとして出力
    try:
        if (not udi_io.input()):
            pass
        elif (udi_io.duel_data != {}):

            # 入力要求に関する情報
            command_request = udi_io.get_command_request()

            # 実行可能なコマンドを取得（内部でcommand_requestを参照しています）
            commands = udi_io.get_commands()

            # デュエルログに関する情報
            #（中間状態などの情報がありますが、このサンプルでは使っていません）
            duel_log_data = udi_io.get_duel_log_data()

            # 管理テーブルを含むデュエルの状況に関する情報
            duel_state_data = udi_io.get_duel_state_data()

            # 管理テーブル本体（duel_state_dataから取得する場合と違い管理テーブルがDuelCardの配列になっています）
            duel_card_table = udi_io.get_duel_card_table()

            # コマンドの履歴
            #command_request.command_logと同じ情報
            #command_log = udi_io.get_command_log()
            
            # 管理テーブル以外を取り出す
            game_data = duel_state_data.general_data
            chain_data = duel_state_data.chain_stack

    except Exception as e:
        print("UdiController.NextStep Error")
        print(e)
        sys.exit()

    #制御用フラグ
    DuelStartFlag = udi_io.is_duel_start()
    DuelEndFlag = udi_io.is_duel_end()
    DuelActionFlag = udi_io.is_command_required()
    
    DuelResult = None
    if(DuelEndFlag):
        #ゲーム完了時の情報
        DuelResult = udi_io.get_duel_end_data()

    #制御情報をパック（ここはAI開発者依存なので、多少分かりづらいですがまとめてます…）
    ControlFlag = [DuelStartFlag, DuelEndFlag, DuelActionFlag, DuelResult]
    Result = [ControlFlag, game_data, duel_card_table, chain_data, command_request, commands]
    return Result

#----------------------------------------------------------------
#アクションを選択して送信
#----------------------------------------------------------------
def SendCommand(index, confidence_score = 0.5, situation_score = 0.5):
    udi_io.output_situation_score([{"score":confidence_score, "player": 0}]) #自己視点でのスコア（キューブ君更新）
    udi_io.output_situation_score([{"score":situation_score, "player": -1}]) #形勢ゲージも一緒に更新
        
    #print("SendComandIndex: %d" % (index))
    udi_io.output_command(index)
    return

#----------------------------------------------------------------
#ステータスベクトル用テーブル
#----------------------------------------------------------------
print("")
print("ステータスベクトル作成用テーブルの設定（デッキ依存）を行います。")
#デッキ依存の設定
#デッキが僅かに異なると、内部テーブルの割り当てが変わるため、AI設計者が管理を行います。
#サンプルでは２枚目のカードとの区別は行いません。

#デッキに使うカードをセット（とりあえず、全部入り）
OneHotTable ={
    1001 : 0 ,
    1002 : 1 ,
    1003 : 2 ,
    1004 : 3 ,
    1005 : 4 ,
    1006 : 5 ,
    1007 : 6 ,
    1008 : 7 ,
    1009 : 8 ,
    1010 : 9 ,
    1011 : 10 ,
    1012 : 11 ,
    1013 : 12 ,
    1014 : 13 ,
    1015 : 14 ,
    1016 : 15 ,
    1017 : 16 ,
    1018 : 17 ,
    1019 : 18 ,
    1020 : 19 ,
    1021 : 20 ,
    1022 : 21 ,
    1023 : 22 ,
    1024 : 23 ,
    1025 : 24 ,
    1026 : 25 ,
    1027 : 26 ,
    1028 : 27 ,
    1029 : 28 ,
    1030 : 29 ,
    1031 : 30 ,
    1032 : 31 ,
    1033 : 32 ,
    1034 : 33 ,
    1035 : 34 
}

TableNum = len(OneHotTable)

#盤面情報用ベクトル数（SetBoardVector関数で使用）
BoardNum = TableNum * constants.PosId.UPPER_VALUE
#ゲーム情報用ベクトル数（SetBoardVector関数で使用）
InforNum = 2 + constants.Phase.UPPER_VALUE + (constants.SelectionType.UPPER_VALUE + 1)
#アクション用ベクトル数（SetActionVector関数で使用）
ActionNum = (constants.CommandType.UPPER_VALUE + 1) + (TableNum + 3)

#----------------------------------------------------------------
#盤面情報セット
#----------------------------------------------------------------
def SetBoardVector(Input): 
    DuelTable = Input[2]
    n = len(DuelTable)

    v = np.zeros(BoardNum + InforNum, dtype=np.float32)

    #自分のカードの盤面情報をセット（このサンプルでは相手の情報は見ません）
    for i in range(n):
        d:models.DuelCard = DuelTable[i]
        if(d.player_id == constants.PlayerId.MYSELF):
            assert d.pos_id >= 0
            #位置情報のみを使用
            #（位置でオフセットをずらし、ワンホット値を設定）
            index = (d.pos_id * TableNum) + OneHotTable[d.card_id]
            v[index] = 1.0

    #基本的なゲーム情報をセット
    GameData = Input[1]
    v[BoardNum + 0] = GameData.lp[0] / 8000.0 #自分のLP
    v[BoardNum + 1] = GameData.lp[1] / 8000.0 #相手のLP
    assert GameData.current_phase >= constants.Phase.DRAW
    assert GameData.current_phase < constants.Phase.UPPER_VALUE
    v[BoardNum + 2 + GameData.current_phase] = 1.0 #フェイズ
    
    #現在の状況をセット
    Request = Input[4]
    index = BoardNum + 2 + constants.Phase.UPPER_VALUE
    v[Request.selection_type + 1 + index] = 1.0

    return v

#----------------------------------------------------------------
#アクション情報セット
#----------------------------------------------------------------
def SetActionVector(Input): 
    ActionData = Input[5]
    n = len(ActionData)

    v = np.zeros((n, ActionNum), dtype=np.float32)
    Offset = constants.CommandType.UPPER_VALUE + 1

    for i in range(n):
        #各アクション情報をセット（サンプルでは一部の情報のみ使用します）
        act = ActionData[i]
        #constants.CommandType.NO_VALUEも加味しておく
        v[i][act.command_type + 1]  = 1.0

        #カード情報をセット
        CardId = act.card_id
        if(CardId == constants.CardId.NO_VALUE):#カードが無い場合（例：フェーズ移動）
            Index = TableNum 
        elif(CardId == constants.CardId.UNKNOWN):#カードが裏（伏せ）の場合
            Index = TableNum + 1 
        elif(CardId in OneHotTable): #カード情報がテーブルにある場合（アクションでは自分が持っていないカードも含まれる可能性があるため注意）
            Index = OneHotTable[act.card_id] #Index < TableNum
        else: #テーブルに含まれないカード（カードプールが決定していれば、別のカードテーブルを作成しておくことは可能）
            Index = TableNum + 2

        #カード情報をセット
        v[i][Offset + Index] = 1.0

    return v

#----------------------------------------------------------------
#DNNモデル定義
#----------------------------------------------------------------
class DnnModel(torch.nn.Module):
    def __init__(self, InputVectorNum = 100) -> None:
        super().__init__()
        self.mLinear0 = torch.nn.Linear(InputVectorNum, 256)
        self.mActvtn0 = torch.nn.ReLU()
        self.mLinear1 = torch.nn.Linear(256, 128)
        self.mActvtn1 = torch.nn.ReLU()
        self.mLinear2 = torch.nn.Linear(128, 64)
        self.mActvtn2 = torch.nn.ReLU()
        self.mLinear3 = torch.nn.Linear(64, 1)
        return
    def forward(self, x):
        y0 = self.mLinear0(x)
        x1 = self.mActvtn0(y0)
        y1 = self.mLinear1(x1)
        x2 = self.mActvtn1(y1)
        y2 = self.mLinear2(x2)
        x3 = self.mActvtn2(y2)
        y3 = self.mLinear3(x3)
        return y3

#----------------------------------------------------------------
#DNNモデル初期化
#----------------------------------------------------------------
print("")
print("ＤＮＮモデルの設定を行います。")
DnnInputNum = BoardNum + InforNum + ActionNum

print("PyTorchのバージョン")
print(torch.__version__)
GpuFlag = torch.cuda.is_available()
Device = 'cpu'
if(GpuFlag):
    Device = 'cuda'

model = DnnModel(DnnInputNum)

#モデルのロード
if(args.LoadWeightName != None):
    print("Loading Weight Data ... %s" % (args.LoadWeightName))
    model.to('cpu')
    model.load_state_dict(torch.load(args.LoadWeightName, weights_only=True))

model.to(Device)
model.eval()

optim = torch.optim.SGD(model.parameters(), lr = 0.0001) #学習率はここで設定
loss = torch.nn.MSELoss()

ReplayMemory = []
ActionMemory = []
LearnCount = 0
LastLoss = 0.0

#----------------------------------------------------------------
#予測
#----------------------------------------------------------------
def DnnPredict(InputData):
    model.eval()
    with torch.no_grad():
        InputTensor = torch.tensor(InputData).to(Device)
        z = model(InputTensor)
        q = z.detach().cpu().numpy()
    return q

#----------------------------------------------------------------
#学習
#----------------------------------------------------------------
def DnnLearn(InputData, LabelData): 
    InputTensor = torch.tensor(InputData).to(Device)
    LabelTensor = torch.tensor(LabelData).to(Device)
    model.train()
    z = model(InputTensor)
    e = loss(z, LabelTensor)
    d = e.detach().cpu().numpy()
    optim.zero_grad()
    e.backward()
    optim.step()

    global LearnCount
    LearnCount += 1
    #print("LearnCount: %d Loss: %f" % (LearnCount, d))

    #自動セーブ
    if(LearnCount % 1000000 == 0):    
        t = datetime.datetime.today()
        s = 'save_%04d%02d%02d_%02d%02d%02d_%04d.pth' % (t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond / 100)
        print("Saving Weight Data ... %s" % (s))
        model.to('cpu')
        torch.save(model.state_dict(), s)
        model.to(Device)

    return d

#----------------------------------------------------------------
#アクション選択
#----------------------------------------------------------------
def SelectAction(commands, UdiData): 
    BatchNum = len(commands)

    #ステータスベクトル作成
    BoardVector = SetBoardVector(UdiData)
    ActionVector = SetActionVector(UdiData)

    #DNN用ベクトルに変換
    #DNNが行うことは、q値 = Q(s, a)を求めることになります。
    #ここでは、状態sを入力として、アクションaを選択するためのq値を求めることになります。
    x = np.empty((BatchNum, DnnInputNum), dtype=np.float32)
    for i in range(BatchNum):
        x[i][0:BoardNum+InforNum] = BoardVector
        x[i][BoardNum+InforNum:DnnInputNum] = ActionVector[i]

    #予測
    OutputValue = DnnPredict(x)

    #最大値を探す
    ActionIndex = random.randrange(BatchNum)
    MaxValue = -999999.9
    for i in range(BatchNum):
        v = OutputValue[i]
        if(v > MaxValue):
            ActionIndex = i
            MaxValue = v
    
    #選択したアクションを保存
    global ActionMemory
    ActionMemory.append(x[ActionIndex])
    return ActionIndex

#----------------------------------------------------------------
#デュエル結果の処理
#----------------------------------------------------------------
def SetResult(DuelResult:models.DuelEndData):
    #報酬決め
    if(DuelResult.result_type == constants.ResultType.WIN):
        Reward = 1.0
    elif(DuelResult.result_type == constants.ResultType.LOSE):
        Reward = -1.0
    else:
        Reward = 0.0

    print("Result: %.1f" % (Reward))

    global ActionMemory
    global ReplayMemory

    #リプレイメモリに保存
    for i in range(len(ActionMemory)):
        ReplayMemory.append([ActionMemory[i], Reward])
    
    #１デュエルのメモリはクリア
    ActionMemory = []
    return

#----------------------------------------------------------------
#学習の更新
#----------------------------------------------------------------
def LearnUpdate():
    global ReplayMemory
    global LastLoss

    #リプレイメモリが一定数貯まったら学習
    n = len(ReplayMemory)
    BatchNum = 32
    
    if(n >= BatchNum):
        x = np.zeros((BatchNum, DnnInputNum), dtype=np.float32)
        y = np.zeros((BatchNum, 1), dtype=np.float32)

        MraxMemorySize = 100000

        for LearnLoop in range(16): #何回か学習する
            #リプレイメモリから取り出す（フォーマット：[状態ベクトル, 報酬]）
            if(n > MraxMemorySize):
                for i in range(BatchNum):
                    index = random.randrange(n)
                    d = ReplayMemory.pop(index) #メモリから削除
                    n -= 1
                    x[i][0:DnnInputNum] = d[0][0:DnnInputNum] #入力ベクトル
                    y[i][0] = d[1] #報酬   
            else:
                for i in range(BatchNum):
                    index = random.randrange(n)
                    d = ReplayMemory[index] #メモリには残す
                    x[i][0:DnnInputNum] = d[0][0:DnnInputNum] #入力ベクトル
                    y[i][0] = d[1] #報酬

            #学習
            #（強化学習アルゴリズムは色々ありますので色々工夫してみてください。）
            #（リプレイメモリを使わない方法もあります。）
            LastLoss = DnnLearn(x, y)
    return

#----------------------------------------------------------------
#デュエル開始時の処理
#----------------------------------------------------------------
def DuelStart(Input):
    DuelTable = Input[2]
    n = len(DuelTable)
    #カード一覧表示
    for i in range(n):
        d:models.DuelCard = DuelTable[i]
        if(d.player_id != -1):
            print("TableIndex: %d, Player: %d, Pos: %s, CardIndex: %d, CardId: %d," % (i, d.player_id, d.pos_id.name, d.card_index, d.card_id))
    return

#----------------------------------------------------------------
#統計情報
#----------------------------------------------------------------
print("統計情報の設定を行います。")
StatisticWin   = 0
StatisticCount = 0
StatisticText = "WinRate 0.000 | WinCount: 0 / 0"
EmaRate = 0.0

#----------------------------------------------------------------
#統計情報更新
#----------------------------------------------------------------
def UpdateStatistic(DuelResult:models.DuelEndData):
    global StatisticWin
    global StatisticCount
    global StatisticText
    global EmaRate
    StatisticCount += 1

    EmaRate *= 0.99
    if(DuelResult.result_type == constants.ResultType.WIN):
        StatisticWin += 1
        EmaRate += 0.01

    StatisticText = "WinRate: %.3f | WinCount: %d / %d | EmaRate: %.3f" % (StatisticWin / StatisticCount, StatisticWin, StatisticCount, EmaRate)
    return

#----------------------------------------------------------------
#現在の状況を表示
#----------------------------------------------------------------
def ShowStatistic():
    #AIが最後に行動した時刻を表示
    t = datetime.datetime.today()
    s = '%04d/%02d/%02d %02d:%02d:%02d' % (t.year, t.month, t.day, t.hour, t.minute, t.second)
    u = "LastLoss = %.3f LearnCount = %d" % (LastLoss, LearnCount)
    x = "★ %s %s @ %s" % (StatisticText, u, s)
    #print("")
    print(x)
    return

#----------------------------------------------------------------
#表示
#----------------------------------------------------------------
print("シミュレーターを起動してください。")
print("例1：CPUとの対戦")
print(r"DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --tcp_port0 50001 --log_level 2")
print("例2：プレイヤー同士との対戦")
print(r"DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1")
print("（停止するときはPythonターミナル上で、Ctrl+Cを押してください）")
print("シミュレーターの起動待ち...")

#----------------------------------------------------------------
#メインループ
#----------------------------------------------------------------
while(True):
    ret = GetNextState()
    control = ret[0]
    if(control[0]):
        print("DuelStart")
        DuelStart(ret)
    elif(control[1]):
        print("DuelEnd")
        UpdateStatistic(control[3])
        if(RandomPlayerFlag != 1):
            SetResult(control[3])

    if(control[2]):
        commands = ret[5]
        if(RandomPlayerFlag == 1):
            # ランダムにコマンド選択
            index = random.randrange(len(commands))  
        else:
            #AIによる予測
            index = SelectAction(commands, ret)

        #コマンドを送信 
        SendCommand(index)
        #学習
        LearnUpdate()
        #統計情報表示
        ShowStatistic()

#終了（※ここには来ない）
print("Done")
