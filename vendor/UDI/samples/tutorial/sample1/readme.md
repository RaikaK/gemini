# 環境設定例

## 仮想環境の設定（初回）

コマンドプロンプトを起動し、下記のコマンドで仮想環境を設定します。

```
cd \Workspace
py -m venv .venv_sample1
.venv_sample1\Scripts\activate
pip install \UDI\libs\ygo
pip install -r \UDI\samples\tutorial\sample1\requirements.txt
```

## 仮想環境の設定（２回目以降）

コマンドプロンプトを起動し、仮想環境を変更します。

```
cd \Workspace
.venv_sample1\Scripts\activate
```

---

# ランダムプレイヤーとの対戦

コマンドプロンプトを２つ起動し、仮想環境を変更します。

サンプルプログラムのフォルダに移動し、pythonスクリプトを起動します。
（tcphostはpythonスクリプトを実行しているPCのIPアドレスになります。）

```
python SampleAi.py -g --tcpport 52010 --tcphost 127.0.0.1
python SampleAi.py -g --tcpport 52011 --tcphost 127.0.0.1 --RandomPlayer 1
```

その後に、DuelSimulatorを起動します。
（tcphost0とtcphost1はpythonスクリプトを実行しているPCのIPアドレスになります。）

```
DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_host0 127.0.0.1 --tcp_port0 52010 --tcp_host1 127.0.0.1 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1
```

学習開始前

```
★ WinRate 0.000 | WinCount: 0 / 0 LastLoss = 0.000 LearnCount = 0 @ 2025/03/12 17:50:10
```

学習開始後

```
★ WinRate: 0.000 | WinCount: 0 / 1 | EmaRate: 0.000 LastLoss = 1.023 LearnCount = 16 @ 2025/03/12 17:52:06
```

WinRateは平均勝率を表します。
EmaRateは直近100～200試合ぐらいの勝率を表します。

何時間か学習させておきますと、WinRateとEmaRateが上がっていきます。




