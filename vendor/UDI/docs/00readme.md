# この文書について

この文書は最初の動作確認の手順のみに絞って説明してます。
それ以外の詳細については docs フォルダ内にある各種ドキュメントをご参照ください。

# 動作確認方法

以下のような手順で動作確認が可能です

1. UDIライブラリの準備
2. サンプルスクリプトの起動
3. MDクライアントでの動作確認
4. Simulatorでの動作確認

## UDIライブラリの準備

1. pythonの準備

    インスタンス2にはpythonをプリインストールしてありますので、まずは作業用の仮想環境を用意します
    (以下では C:\Workspace 以下に作成します)
    コマンドプロンプトを立ち上げ
    ```
    > cd \Workspace
    > py -m venv .venv
    > .venv\Scripts\activate
    ```
    と順に実行します

2. UDIライブラリのインストール

    C:\UDI\libs\ygo フォルダにライブラリがありますので仮想環境にインストールします
    ```
    > pip install \UDI\libs\ygo
    ```
    みたいな感じになります

## サンプルスクリプトの起動

C:\UDI\samples\basic フォルダに基本的な機能を持ったサンプルスクリプトが入ってます

- udi_simple_v1.py ... UDIを使用したAIスクリプトの最小構成なサンプル
- udi_random_v1.py ... 選択可能なコマンドからランダム返答するサンプル(udi_simple_v1.pyをベースにライブラリの機能を盛っていったもの)
- udi_random_vs_random_v1.py ... ランダム返答同士でデュエルするサンプル(udi_random_v1.pyをベースにAI vs AIの形にしたもの)
- udi_check_commands_v1.py ... キーボード操作でコマンド手入力するサンプル
- udi_gui_sample.py ... GUIベースでSimulatorなどを操作するサンプル

ここではまず udi_random_v1.py を使用することとします。コンソールから
```
> python \UDI\samples\basic\udi_random_v1.py --tcp_host インスタンス2のIPアドレス
Start Socket Server: インスタンス2のIPアドレス:50001
connect to client ...
```
みたいな感じで実行し、クライアント(MDクライアントもしくはSimulator)の接続を待つ状態になっていればひとまず成功です

## MDクライアントでの動作確認

以下の手順でMDクライアントでの動作確認が可能です

1. インスタンス1にDCVで接続
2. C:\UDI\MD1 フォルダ内の masterduel.exe をエクスプローラからダブルクリックして起動
3. C:\UDI\MD2 フォルダ内の masterduel.exe をエクスプローラからダブルクリックして起動
4. 画面の指示に従ってホーム画面まで進める(言語は日本語、地域は日本を選んでください)
5. インスタンス2側で2つ目のコンソールを立ち上げ、もう1個ポート違いで udi_random_v1.py を実行
    ```
    > cd \Workspace
    > py -m venv .venv
    > .venv\Scripts\activate
    > python \UDI\samples\basic\udi_random_v1.py --tcp_host インスタンス2のIPアドレス --tcp_port 50002
    Start Socket Server: インスタンス2のIPアドレス:50002
    connect to client ...
     ```
6. インスタンス1で立ち上げたMDで「DUEL」に進んでルーム戦を開始します(片方はルームの作成、もう一方はルームに入るの検索から)
7. インスタンス2の udi_random_v1.py を実行しているコンソールに
    ```
    Start Socket Server: インスタンス2のIPアドレス:50001
    connect to client ...
    done.
    response: udi_length: ??????
    response: Ok
    DuelStart - card: {...
    ```
    みたいな感じで「done.」以降が流れてれば無事動作してます。

## Simulatorでの動作確認

以下の手順でSimulatorでの動作確認が可能です

1. インスタンス1で実行しているMDクライアントを終了させておきます
2. インスタンス2の udi_random_v1.py を実行しているコンソールで Ctrl-C 等で一旦終了させてから再度実行しておきます(再度実行は最初に立ち上げたほうのみでOK)
3. インスタンス1の C:\UDI\Simulator1 フォルダ内の DuelSimulator.exe をエクスプローラからダブルクリックして起動
4. udi_random_v1.py を実行しているコンソールにログが流れてれば無事動作してます

# サンプルなどを編集したい場合

サンプルなどが置いてある、インスタンス2の C:\UDI フォルダにあるファイルはそのままでは上書き出来ません。また、運営側でバージョンアップなどを行う際にフォルダ内丸ごと上書きする場合があります。よって、編集などを行う際には別フォルダに一旦コピーしてください

MDクライアントやSimulatorが置いてある、インスタンス1の C:\UDI フォルダでは、設定ファイル(UdiSettings.json および DuelSimulator.json)は上書き可能にしてあります。インスタンス2と同様に運営側でバージョンアップなどを行う際にフォルダ内丸ごと上書きする場合がありますが、設定ファイルとMDクライアントのセーブデータなどについては基本残すようにする予定です