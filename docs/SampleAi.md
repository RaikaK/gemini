# SampleAi.py コード解析

## 1. メインループの構造

AI の思考と行動を制御するメインループは、以下のコードで構成されている。

```python
while True:
    ret = GetNextState()
    control = ret[0]

    if control[0]:
        print("DuelStart")
        DuelStart(ret)

    elif control[1]:
        print("DuelEnd")
        UpdateStatistic(control[3])

        if RandomPlayerFlag != 1:
            SetResult(control[3])

    if control[2]:
        commands = ret[5]

        if RandomPlayerFlag == 1:
            index = random.randrange(len(commands))

        else:
            index = SelectAction(commands, ret)

        SendCommand(index)
        LearnUpdate()
        ShowStatistic()
```

1.  **状態取得:** `GetNextState()` を呼び出し、シミュレータから現在のゲーム状態（`ret`）を取得する
2.  **デュエル開始/終了処理:**
    - デュエル開始時 (`control[0] == True`): `DuelStart()` を実行する
    - デュエル終了時 (`control[1] == True`): `UpdateStatistic()` で統計を更新し、AI プレイヤーの場合のみ `SetResult()` で結果を処理する
3.  **アクション要求処理:**
    - 行動選択が必要な場合 (`control[2] == True`):
      - **行動選択:** ランダムプレイヤーか AI かに応じて、`random` または `SelectAction()` で実行するコマンドのインデックス (`index`) を決定する
      - **コマンド送信:** `SendCommand(index)` で決定した行動をシミュレータに送信する
      - **学習:** `LearnUpdate()` を呼び出し、リプレイメモリを用いてモデルの学習を行う
      - **統計表示:** `ShowStatistic()` で現在の学習状況や勝率を表示する

## 2. 主要な変数の定義

状態や行動をベクトル化するために、以下の変数が定義されている。

- `OneHotTable`: デッキに含まれるカード ID を、ベクトルのインデックスに対応付ける辞書
- `TableNum`: カードの種類数 (`len(OneHotTable)`)
- `BoardNum`: **盤面状態ベクトル**の次元数
  - 計算式: `TableNum * 「カードの位置を表す変数の上限値」`
  - 意味: 「どのカードが」「どの場所に」存在するかを表現するパターン数
- `InforNum`: **ゲーム情報ベクトル**の次元数
  - 計算式: `2 (不明) + 「フェーズを表す変数の上限値」 + (「コマンド入力のタイミングを表す変数の上限値」 + 1（不明）)`
  - 意味: いつかを表現するパターン数
- `ActionNum`: **単一の行動ベクトル**の次元数
  - 計算式: `(行動の種類を表す変数の上限値 + 1) + (TableNum + 3（不明）)`
  - 意味: コマンドの種類と、関連するカードを表現するパターン数
- `DnnInputNum`: DNN モデルへの総入力次元数 (`BoardNum + InforNum + ActionNum`)
- `ActionMemory`: 1 デュエル（1 エピソード）で選択した**行動の INDEX たち**を一時保存するリスト
- `ReplayMemory`: 全てのデュエルで蓄積された、**[ActionMemory, 報酬]** のペアを格納するリスト

## 3. 各関数の役割

- **`GetNextState()`**
  シミュレータから現在のデュエル状態を包括的に取得し、リスト形式で返す。

  ```python
  [
      # [0]: 制御フラグ [is_duel_start, is_duel_end, is_command_required, duel_result]
      control,
      # [1]: LPやフェーズなどの一般情報 (GeneralData)
      game_data,
      # [2]: 全カードの配置情報 (List[DuelCard])
      duel_card_table,
      # [3]: 現在のチェーン情報 (ChainStack)
      chain_data,
      # [4]: 入力要求の詳細 (CommandRequest)
      command_request,
      # [5]: 実行可能な全コマンドのリスト (List[CommandEntry])
      commands
  ]
  ```

- **`DuelStart(ret)`**
  デュエル開始時に、初期盤面のカード情報（`duel_card_table`）をコンソールに`print`する。

- **`UpdateStatistic(ret[0][3])`**
  デュエルの結果に基づきグローバル変数を更新する。

  ```python
  global StatisticWin # 勝利数
  global StatisticCount # デュエル数
  global StatisticText # 統計情報文字列
  global EmaRate # 指数移動平均勝率
  ```

- **`SetResult(ret[0][3])`**

  1.  デュエルの勝敗（WIN/LOSE/DRAW）に応じて、報酬 `Reward` を `1.0 / -1.0 / 0.0` に設定して、`print`する
  2.  `ActionMemory` に保存されたそのデュエル中の全行動に対し、この報酬を紐付け、`ReplayMemory` に追加する

- **`SelectAction(ret[5], ret)`**

  1.  `SetBoardVector()` と `SetActionVector()` を呼び出し、現在の「状態」と「各行動」をベクトル化する
  2.  全ての「状態＋行動」の組み合わせベクトル（サイズ：`選択可能なコマンドの数`\*(`SetBoardVectorのサイズ`+`SetActionVectorのサイズ`)を DNN に入力し、それぞれの行動価値（Q 値）を `DnnPredict()` で予測する
  3.  予測された価値が最大となる行動を選択し、その行動の**INDEX**を`ActionMemory`に保存する
  4.  価値最大の行動の**INDEX**を返す。

- **`SetBoardVector(ret)`**
  現在の盤面情報(`duel_card_table`)、ゲーム情報(`game_data`)、入力要求(`command_request`)を一つの長い状態ベクトルに符号化して返す。（サイズ：`BoardNum` + `InforNum`）

  > **改善コメント:**
  > 現在の実装では `chain_data` が考慮されていない。`DuelStateData`（`game_data`, `duel_card_table`, `chain_stack`）と `command_request` をいかに効果的に符号化するかが、AI の性能向上の鍵となる。

- **`SetActionVector(ret)`**
  選択肢として提示された全コマンド(`commands`)を、それぞれ行動ベクトルに符号化し、2 次元配列として返す。（サイズ：`選択可能なコマンドの数`\*`ActionNum`）

  > **改善点の指摘:** > `CommandEntry` の符号化を行えば良い。

- **`SendCommand(SelectActionが返したINDEX)`**
  選択した行動のインデックスを、自己評価スコアと共にシミュレータへ送信する。

- **`LearnUpdate()`**
  `ReplayMemory` に蓄積されたデータがバッチサイズ以上ある場合、ランダムサンプリングを行い、`DnnLearn()` を用いてモデルの学習を 16 回実行する。メモリが上限を超えている場合は、古いデータから削除しながらサンプリングする。

- **`DnnLearn(入力ベクトル, 報酬)`**
  入力ベクトルから現在のモデルで行動価値を再計算する。

  1.  モデルの出力と教師データ（デュエルの結果から得られた報酬）との間で平均二乗誤差（MSE）を計算する
  2.  誤差を逆伝播させ、モデルの重みを更新（最適化）する
  3.  一定の学習回数ごとに、モデルの重みをファイルに保存する

- **`ShowStatistic()`**
  現在の勝率、学習回数、最新の損失（Loss）などの`StatisticText`を`print`する。
