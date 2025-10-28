# 🐉 教師データ収集ガイド

これは、2 台のリモート Windows インスタンスを使用し、**GUI** と **Simulator** を起動して、**教師データ**を収集する手順です。後々、**AI** が教師データを使用して学習します。

---

## 1. インスタンス 2 : GUI の起動

1. **Windows PowerShell** を開きます。（既に開いているものでも可。）
2. 以下のコマンドを順番に実行し、**GUI** を起動します。

   ```shell
   cd C:\Users\b1\Desktop\u-ni-yo\
   .\.u-ni-yo\Scripts\activate
   git checkout main
   git pull
   ```

   （Note : `git`操作が失敗する場合は、誰かのコード変更が残っている可能性があります。`git stash`を実行し、コード変更を一時退避させてください。）

   ```shell
   python .\scripts\collect_human_data.py
   ```

3. GUI が表示されたら、ウィンドウを**全画面表示**にしてください。これで、GUI の起動は完了です。

---

## 2. インスタンス 1 : Simulator の起動

1. **コマンドプロンプト**を開きます。（既に「Simlator1」、「Simlator2」もしくは「Simlator3」とタイトル付けされたコマンドプロンプトが開いていたら、空いているものを選んでください。）
2. 以下のコマンドを実行し、**Simulator** を起動します。

   ```shell
   DuelSimulator.exe --player_type0 Human --player_type1 CPU --deck_path0 DeckData/RoyaleBE.json --deck_path1 DeckData/RoyaleBE.json --first_player 0 --lp0 8000 --lp1 8000 --hand_num0 5 --hand_num1 5 --log_level 1 --loop_num 100000 --randomize_seed true --play_reverse_duel true --exit_with_udi true --connect gRPC --grpc_deadline_seconds 60 --tcp_host0 10.95.102.79 --tcp_port0 53000 --tcp_host1 10.95.102.79 --tcp_port1 53100 --on_start_retry_connect_seconds 60
   ```

3. **インスタンス 2 の GUI**に戻ってください。ゲームが開始されているはずです。これで、Simulator の起動は完了です。ひたすら戦ってください。最大で、連続 100,000 試合戦うことができます。

---

## 3. GUI についての補足

- **データ保存** : 1 試合が終了するごとに、その試合の記録が 1 つのファイルとして、自動で保存されます。（試合を中断した場合は、その試合の記録は残りません。）
- **試合数** : これまでに保存された累積試合数は、画面中央上部に「何試合目」といった形で表示されています。
- **画面調整** : 枠が見切れたり、枠のバランスがおかしい場合は、画面右上の「拡大」「縮小」ボタンで表示サイズを調整してください。
- **デバッグモード** : 画面右上の「デバック」チェックボックスをオンにすると、コマンドなどの詳細情報（`CommandEntry`の中身など）が見えます。基本的には、オフ（チェックしない）の方が操作しやすいです。

---

## 4. 終了処理

以下の手順で各プロセスを停止します。

1. **インスタンス 2** : GUI を画面右上の`×`ボタンで消します。
2. **インスタンス 2** : Windows PowerShell で`Ctrl`+`C`を押し、GUI を停止します。
3. **インスタンス 1** : コマンドプロンプトで`Ctrl`+`C`を押し、Simulator を停止します。（GUI を停止すると、自動で Simulator も停止する場合があります。）
