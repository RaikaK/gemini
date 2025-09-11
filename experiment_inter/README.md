# 面接シミュレーション実験システム

## 概要

このプロジェクトは、AI面接官とAI学生による面接シミュレーション実験を行うシステム。
面接官役のLLMが学生の志望度を正確に判定できるかを評価し、面接フローの最適化を研究するためのツール。

## 主な機能

### 1. 面接シミュレーション
- **AI面接官**: ローカルモデル（Llama等）またはAPIモデル（GPT等）を使用
- **AI学生**: 異なる準備レベル（high/medium/low）を持つ候補者をシミュレート
- **動的面接フロー**: 面接の進行に応じて質問タイプを智的に決定

### 2. 評価システム
- **評価1**: 最も志望度が低い候補者の選定精度
- **評価2**: 候補者の志望度ランキング精度
- **評価3**: 知識欠損検出精度

### 3. モデル管理
- **ローカルモデル**: Hugging Face CLIを使用したモデル管理
- **APIモデル**: OpenAI APIを使用したモデル選択
- **量子化対応**: 4bit量子化によるメモリ効率化

### 4. データ管理
- **実験結果保存**: JSON形式での詳細な結果記録
- **スプレッドシート連携**: Google Apps Scriptを使用した結果の自動記録
- **過去結果閲覧**: Webインターフェースでの結果確認

## システム構成

```
experiment_inter/
├── app.py                    # Flask Webアプリケーション（メイン）
├── interv.py                 # 面接官クラス（Interviewer）
├── student.py                # 学生クラス（GPTApplicant）
├── model_manager.py          # ローカルモデル管理
├── spreadsheet_integration.py # スプレッドシート連携
├── data_generators.py        # データ生成・読み込み
├── config.py                 # 設定ファイル
├── utils.py                  # ユーティリティ関数
├── db.json                   # 実験データベース
├── spreadsheet_config.json   # スプレッドシート設定
├── templates/
│   └── index.html           # Webインターフェース
└── results/                 # 実験結果保存ディレクトリ
```

## 起動方法

### 1. 依存関係のインストール
```bash
uv pip install flask openai transformers torch requests
```

### 2. 設定ファイルの編集
`config.py`でAPIキーとモデル設定を編集：
```python
OPENAI_API_KEY = "your-api-key-here"
INTERVIEWER_MODEL_TYPE = 'api'  # 'local' or 'api'
```

### 3. アプリケーションの起動
```bash
python app.py
```

### 4. Webインターフェースにアクセス
ブラウザで `http://localhost:5000` にアクセス

## 実験の流れ

1. **データ準備**: `db.json`から企業情報と学生プロフィールを読み込み
2. **面接実行**: 面接官が学生に質問し、回答を収集
3. **評価**: 3つの評価タスクを実行
4. **結果保存**: JSONファイルとスプレッドシートに結果を記録

## 設定オプション

### 面接フロー設定
- **固定フロー**: 事前定義された質問パターン
- **動的フロー**: AIが状況に応じて質問タイプを決定

### モデル選択
- **ローカルモデル**: Llama 3.1, ELYZA, SWALLOW等
- **APIモデル**: GPT-4o, GPT-4o-mini等

### 実験パラメータ
- 候補者数（デフォルト: 3人）
- 最大面接ラウンド数
- 面接フローの種類

## 結果の確認

### Webインターフェース
- リアルタイムでの実験進捗確認
- 過去の実験結果一覧
- 詳細な結果データの表示

### ファイル出力
- `results/`ディレクトリにJSON形式で保存
- 個別シミュレーション結果と全体サマリー

### スプレッドシート連携
- Google Apps Scriptを使用した自動記録
- 実験結果の集計と分析

## 技術仕様

### 使用技術
- **Backend**: Python, Flask
- **AI Models**: OpenAI API, Hugging Face Transformers
- **Frontend**: HTML, JavaScript, Bootstrap
- **Data Storage**: JSON, Google Sheets

### 対応モデル
- **ローカル**: Llama 3.1, ELYZA, SWALLOW, CALM2等
- **API**: GPT-4o, GPT-4o-mini, GPT-4等

## 注意事項

1. **APIキー**: OpenAI APIキーの設定が必要
2. **ローカルモデル**: 初回使用時にダウンロードが必要（数GB〜数十GB）
3. **GPU**: ローカルモデル使用時はGPU推奨
4. **スプレッドシート**: 連携機能はオプション（設定不要でも動作）

## 更新履歴

- v1.0: 基本的な面接シミュレーション機能
- v1.1: 動的面接フロー機能追加
- v1.2: スプレッドシート連携機能追加
- v1.3: ローカルモデル管理機能強化
