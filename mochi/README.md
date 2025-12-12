# 面接ロールプレイ実行システム（最小限版）

## 概要

LLM同士が面接のやりとりを行い、3人の候補者の中で面接ロールプレイを実行する最小限のシステムです。
WebGUIを排除し、必要最小限の機能のみを実装しています。

## クイックスタート

実際の面接シミュレーションを実行:

```bash
cd mochi
export OPENAI_API_KEY='your-api-key-here'
pip install -r requirements.txt
python main.py
```

APIプロバイダーを指定して実行 (デフォルトはconfig.pyの設定、またはgoogle):

```bash
# Google API (Gemini) を使用
python main.py --api-provider google

# OpenAI API を使用
python main.py --api-provider openai
```

複数回のシミュレーションを実行:

```bash
# 5回実行
python main.py -n 5

# 特定のデータセットを使用して3回実行
python main.py -n 3 -s 0
```

## 特徴

- **シンプルな構成**: WebGUIなし、CLIベースで実行
- **3者間ロールプレイ**: 面接官1名 + 応募者3名でシミュレーション
- **マルチAPI対応**: OpenAI API と Google API (Gemini) の切り替えが可能
- **ローカルモデル対応**: ローカルLLMを使用したシミュレーションも可能
- **複数回シミュレーション対応**: コマンドライン引数で実行回数を指定可能
- **評価2の精度計算**: ランキング予測の精度を自動計算（正しく抽出できた場合のみ）
- **最小限の依存**: 必要なライブラリを最小化

## ファイル構成

```
mochi/
├── config.py                  # 設定ファイル（APIキー、モデル名など）
├── utils.py                   # ユーティリティ関数（API呼び出し）
├── interviewer.py             # 面接官役のクラス
├── student.py                 # 応募者役のクラス
├── metrics.py                 # 評価メトリクス計算
├── runner.py                  # 面接フロー実行ロジック
├── main.py                    # CLIエントリ
├── requirements.txt           # 依存ライブラリ
├── results/                   # 結果保存ディレクトリ（自動生成）
└── README.md                  # このファイル
```

## セットアップ

1. **依存ライブラリのインストール**
   ```bash
   pip install -r requirements.txt
   ```
   
2. **APIキーの設定**
   
   **Mac/Linux (bash/zsh):**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:OPENAI_API_KEY="your-api-key-here"
   ```

   **Windows (Command Prompt):**
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```
   
   方法2: `local_config.py`を作成（推奨・Git管理外）
   プロジェクトルートに `local_config.py` を作成し、以下のように記述します。このファイルはGitにコミットされません。
   （テンプレート `local_config.example.py` をコピーして使用できます）
   ```python
   OPENAI_API_KEY = "your-api-key-here"
   # 必要に応じて他も上書き可能
   # API_PROVIDER = "openai"
   ```

   方法3: config.pyを直接編集（非推奨）
   ```python
   OPENAI_API_KEY = "your-api-key-here"
   ```

3. **Google APIキーの設定 (Gemini使用時)**
   
   **Mac/Linux (bash/zsh):**
   ```bash
   export GOOGLE_API_KEY="your-google-api-key-here"
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:GOOGLE_API_KEY="your-google-api-key-here"
   ```

   **Windows (Command Prompt):**
   ```cmd
   set GOOGLE_API_KEY=your-google-api-key-here
   ```

   または `local_config.py` に記述:
   ```python
   GOOGLE_API_KEY = "your-google-api-key-here"
   API_PROVIDER = "google"
   ```
   

## 使い方

### 基本的な実行

```bash
cd mochi
python main.py
```

### 複数回シミュレーション

```bash
# 5回実行
python main.py -n 5

# または
python main.py --num-simulations 5

# 特定のデータセットを指定して3回実行
python main.py -n 3 -s 0

# APIプロバイダーを指定
python main.py --api-provider google
```

### 動作の流れ

1. `experiment_inter/db.json`からランダムに企業情報と候補者情報を読み込む
2. 3名の候補者をそれぞれ異なる準備レベル（high/medium/low）で初期化
3. 面接官が質問を生成し、各候補者が回答
4. これを指定ラウンド数（デフォルト5回）繰り返す
5. 最終評価として、最も志望度が低い候補者を選定し、ランキングを作成
6. 結果を`results/`ディレクトリにJSON形式で保存

### 設定のカスタマイズ

`config.py`で以下の設定を変更できます：

- `NUM_CANDIDATES`: 候補者数（デフォルト: 3）
- `MAX_ROUNDS`: 面接ラウンド数（デフォルト: 5）
- `INTERVIEWER_MODEL`: 面接官のモデル名（デフォルト: "gemini-2.5-flash-lite"）
- `APPLICANT_MODEL`: 応募者のモデル名（デフォルト: "gemini-2.5-flash-lite"）
- `API_PROVIDER`: 使用するAPIプロバイダー ("google" または "openai")
- `NUM_SIMULATIONS`: デフォルトのシミュレーション実行回数（デフォルト: 1）

## 出力例

実行すると以下のような出力が得られます：

```
============================================================
面接ロールプレイ実行システム（最小限版）
============================================================

=== セット 1 を選択 ===
企業: サンプル株式会社
学生数: 3人

候補者: 山田太郎 (準備レベル: high, 知識カバレッジ: 10/10 (100%))
候補者: 佐藤花子 (準備レベル: medium, 知識カバレッジ: 7/10 (70%))
候補者: 鈴木次郎 (準備レベル: low, 知識カバレッジ: 4/10 (40%))

============================================================
面接開始（5ラウンド）
============================================================

--- ラウンド 1 ---
面接官の質問: 当社のビジョンについてどのように理解していますか？

山田太郎: 御社のビジョンは...（回答）
佐藤花子: 御社のビジョンについては...（回答）
鈴木次郎: 御社のビジョンに関して...（回答）

...（以降のラウンド）

============================================================
最終評価
============================================================

【最も志望度が低い候補者の選定】
最も志望度が低いのは鈴木次郎さんです。理由：...

【志望度ランキング（低い順）】
1位: 鈴木次郎
2位: 佐藤花子
3位: 山田太郎

【評価2: ランキング精度】
  精度スコア: 1.000
  正解ペア数: 3/3
  完全一致位置数: 3/3

============================================================
結果を保存しました: results/interview_result_sim1_20250122_120000.json
============================================================
```

## 元のプログラムとの違い

元の`experiment_inter/app.py`との主な違い：

- **WebGUI削除**: Flask、テンプレート、静的ファイルなし
- **ローカルモデル対応**: APIだけでなくローカルLLMもサポート
- **マルチAPI対応**: OpenAIに加えGoogle (Gemini) もサポート
- **複数回シミュレーション**: コマンドライン引数で実行回数を指定可能
- **評価2の精度計算**: ランキング予測の精度を自動計算（正しく抽出できた場合のみ）
- **統計分析削除**: 基本的な評価のみ
- **人間面接官モード削除**: LLM同士のみ
- **動的フロー削除**: 固定ラウンド数のみ

## トラブルシューティング

### `db.json`が見つからない

`mochi/`ディレクトリは`experiment_inter/`と同じ階層にある必要があります。
プロジェクト構造：
```
penguin-paper/
├── experiment_inter/
│   └── db.json
└── mochi/
    └── main.py
```
