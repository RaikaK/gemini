# Agent

## 実装手順
`src\agents\base_ygo_agent.py`内のBaseYgoAgentを継承して、

* `__init__(self, ...)`
* `select_action(self, state:dict)`
* `update(self, state: dict, action_data: ActionData, next_state: dict)`

を実装すること

この3つを実装すれば、`player1_thread.py`内の`agent`インスタンスを書き換えるだけで学習やテストプレイ等できるはず


## `__init__()`メソッド
エージェントのコンストラクタ

DNNモデルの初期化など、必要なメンバ変数やモジュールの初期化をここで行ってください。


## `select_action()`メソッド

引数: stateを受け取り、行動データActionDataを返してください

```python
state: dict = {
    "is_duel_start": is_duel_start, 
    # デュエル開始の合図 YgoEnv.step()内で呼ばれる
    # おそらくエージェントが受け取るときはfalseとなっているはず 
    # -> 必要であればstateとして返してもよいが、相談しましょう
    "is_duel_end": is_duel_end, 
    # デュエル終了の合図doneとしての役割
    "is_cmd_required": is_cmd_required,
     # コマンド入力が必要な時 | is_duel_end==Falseの時は、常にTrueを返す
    "duel_end_data": duel_end_data: DuelEndData, 
    # デュエル終了時の結果等が含まれている。Win or Loseなど
    "state": state: DuelStateData, 
    # デュエルの状態を持つデータ
    "command_request": command_request: CommandRequest, 
    # 入力可能なコマンドリストなどを含むデータ
    "reward": reward: float, 
    # 勝ち: 1 | 負け: -1 | その他: 0
}

# 現在の状態"state"の時の行動データActionをデータとして管理したいので、このようなつくりにしてます。
action_data = ActionData(
    state: DuelStateData,
    # state["state"]をそのまま引数としてください
    command_request: CommandRequest, 
    # state["state"]時、要求されているコマンドリストなどを含むデータ
    # どのようなコマンドが可能な時に、何を選んだかを反映させるために、引数として求めています。
    command_entry: CommandEntry
    # 実際に選んだコマンド
    # command_requestとcommand_entryの結果をもとに、UdiIOに送信されるコマンドインデックスが内部メソッドで決定されます。
)
```



## `update()`メソッド

引数として、
* 現在の状態 | state: dict
* アクション | action_data: ActionData | 状態state時に選択した行動
* 次状態 | next_state: dict | 現在の状態stateの時、選択した行動action_dataをした結果

を受け取り、エージェントの学習や内部メモリへの経験データ蓄積などを行う

帰り値は、None or log_dictとしてください。

Noneの時は、ログスキップするように`player0_thread.py`に書いています。
ログなどの情報は、

```python
log_dict = agent.update(...)
if log_dict is not None:
    wandb.log(log_dict) 
    # wandb.log()は辞書型のみ受け付けるのでこのようにしています

```



