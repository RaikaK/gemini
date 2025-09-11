# 機能詳細ドキュメント

## 目次
1. [app.py - メインアプリケーション](#apppy---メインアプリケーション)
2. [interv.py - 面接官クラス](#intervpy---面接官クラス)
3. [student.py - 学生クラス](#studentpy---学生クラス)
4. [model_manager.py - モデル管理](#model_managerpy---モデル管理)
5. [spreadsheet_integration.py - スプレッドシート連携](#spreadsheet_integrationpy---スプレッドシート連携)
6. [data_generators.py - データ生成](#data_generatorspy---データ生成)
7. [config.py - 設定ファイル](#configpy---設定ファイル)
8. [utils.py - ユーティリティ](#utilspy---ユーティリティ)

---

## app.py - メインアプリケーション

Flask Webアプリケーションのメインファイル。面接シミュレーションの実行、結果管理、Webインターフェースを提供。

### 主要なグローバル変数
- `experiment_status`: 実験の実行状態を管理
- `experiment_queue`: 実験実行用のキュー
- `model_manager`: Hugging Faceモデル管理インスタンス

### 主要関数

#### `log_message(message)`
- **機能**: ログメッセージを追加
- **引数**: `message` (str) - ログメッセージ
- **処理**: タイムスタンプ付きでログを記録（最大100件保持）

#### `update_progress(progress, step)`
- **機能**: 実験の進捗状況を更新
- **引数**: 
  - `progress` (int) - 進捗率（0-100）
  - `step` (str) - 現在のステップ名

#### `calculate_accuracy_metrics(candidate_states, least_motivated_eval, ranking_eval, knowledge_gaps_eval)`
- **機能**: 面接官の評価精度を計算
- **引数**:
  - `candidate_states`: 候補者の状態データ
  - `least_motivated_eval`: 評価1の結果
  - `ranking_eval`: 評価2の結果
  - `knowledge_gaps_eval`: 評価3の結果
- **戻り値**: 精度指標（accuracy, precision, recall, f1_score等）

#### `calculate_ranking_accuracy(candidate_states, ranking_eval)`
- **機能**: ランキング評価の精度を計算
- **引数**:
  - `candidate_states`: 候補者の状態データ
  - `ranking_eval`: ランキング評価結果
- **戻り値**: ランキング精度指標

#### `initialize_local_model(model_name)`
- **機能**: ローカルモデルの初期化
- **引数**: `model_name` (str) - モデル名
- **処理**: 
  - Hugging Face CLIの確認・インストール
  - モデルのダウンロード（初回時）
  - モデルの初期化
- **戻り値**: `(model, tokenizer)` タプル

#### `run_single_experiment(...)`
- **機能**: 単一の面接シミュレーションを実行
- **引数**:
  - `local_interviewer_model`: ローカル面接官モデル
  - `local_interviewer_tokenizer`: トークナイザー
  - `set_index`: データセットインデックス
  - `simulation_num`: シミュレーション番号
  - `interview_flow`: 面接フロー
  - `use_dynamic_flow`: 動的フロー使用フラグ
  - `interviewer_model_type`: 面接官モデルタイプ
  - `interviewer_model_name`: 面接官モデル名
- **処理**:
  1. データ読み込み
  2. 面接官・候補者の初期化
  3. 面接フローの実行
  4. 最終評価
  5. 精度指標の計算
- **戻り値**: 実験結果辞書

#### `run_experiment_web(...)`
- **機能**: Web用の面接シミュレーション実行（複数回対応）
- **引数**: `run_single_experiment`と同様
- **処理**:
  - 複数シミュレーションの実行
  - 結果の集計
  - ファイル保存
  - スプレッドシート記録

#### `get_experiment_results()`
- **機能**: 過去の実験結果一覧を取得
- **戻り値**: 結果ファイルのリスト

### Web API エンドポイント

#### `@app.route('/')`
- **機能**: メインページの表示
- **戻り値**: `index.html`テンプレート

#### `@app.route('/api/start_experiment', methods=['POST'])`
- **機能**: 実験の開始
- **引数**: JSON形式の実験設定
- **処理**: バックグラウンドで実験を実行

#### `@app.route('/api/status')`
- **機能**: 実験の状態を取得
- **戻り値**: 現在の実験状態

#### `@app.route('/api/results')`
- **機能**: 過去の実験結果一覧を取得
- **戻り値**: 結果ファイルのリスト

#### `@app.route('/api/results/<filename>')`
- **機能**: 特定の実験結果の詳細を取得
- **引数**: `filename` - 結果ファイル名
- **戻り値**: 実験結果の詳細データ

#### `@app.route('/api/models/status')`
- **機能**: ローカルモデルの状態を取得
- **戻り値**: モデルのダウンロード状況とディスク使用量

#### `@app.route('/api/models/download', methods=['POST'])`
- **機能**: モデルのダウンロード
- **引数**: `model_key` - ダウンロードするモデルのキー

#### `@app.route('/api/models/cleanup', methods=['POST'])`
- **機能**: モデルの削除
- **引数**: `model_key` - 削除するモデルのキー

#### `@app.route('/api/spreadsheet/status')`
- **機能**: スプレッドシート連携の状態を取得
- **戻り値**: 接続状況と統計情報

#### `@app.route('/api/spreadsheet/initialize', methods=['POST'])`
- **機能**: スプレッドシートの初期化

#### `@app.route('/api/spreadsheet/record', methods=['POST'])`
- **機能**: 実験結果をスプレッドシートに記録

#### `@app.route('/api/spreadsheet/clear', methods=['POST'])`
- **機能**: スプレッドシートのデータをクリア

---

## interv.py - 面接官クラス

面接官役のLLMを扱う統合クラス。ローカルモデルとAPIモデルの両方に対応。

### クラス: `Interviewer`

#### `__init__(company_profile, model_type, model=None, tokenizer=None)`
- **機能**: 面接官の初期化
- **引数**:
  - `company_profile` (dict): 企業情報
  - `model_type` (str): 'local' または 'api'
  - `model`: ローカルモデル（オプション）
  - `tokenizer`: トークナイザー（オプション）

#### `_generate_response(prompt, max_tokens=512)`
- **機能**: モデルタイプに応じて応答を生成
- **引数**:
  - `prompt` (str): プロンプト
  - `max_tokens` (int): 最大トークン数
- **処理**: ローカルモデルまたはAPIモデルで応答生成
- **戻り値**: 生成された応答テキスト

#### `ask_common_question(all_questions_history)`
- **機能**: 全候補者に対する共通質問を生成
- **引数**: `all_questions_history` (list) - 過去の質問履歴
- **処理**: 戦略的な全体質問を生成
- **戻り値**: `(question, thought)` タプル

#### `ask_question(conversation_history)`
- **機能**: 個別候補者に対する質問を生成
- **引数**: `conversation_history` (list) - 会話履歴
- **処理**: 未言及項目に基づく個別質問を生成
- **戻り値**: `(question, thought)` タプル

#### `should_continue_interview(conversation_history, current_round, max_rounds=5)`
- **機能**: 面接の継続判断
- **引数**:
  - `conversation_history`: 会話履歴
  - `current_round`: 現在のラウンド
  - `max_rounds`: 最大ラウンド数
- **戻り値**: `(should_continue, reason)` タプル

#### `decide_next_question_type(candidate_states, asked_common_questions, current_round, max_rounds=5)`
- **機能**: 次の質問タイプ（全体質問 vs 個別質問）を決定
- **引数**:
  - `candidate_states`: 候補者の状態
  - `asked_common_questions`: 実施済み全体質問
  - `current_round`: 現在のラウンド
  - `max_rounds`: 最大ラウンド数
- **戻り値**: `(question_type, reason)` タプル

#### `conduct_dynamic_interview(candidate_states, applicant, max_rounds=10)`
- **機能**: 智的な動的面接フローを実行
- **引数**:
  - `candidate_states`: 候補者の状態
  - `applicant`: 学生役のインスタンス
  - `max_rounds`: 最大ラウンド数
- **処理**: 状況に応じて質問タイプを動的に決定
- **戻り値**: `(total_rounds, actual_interview_flow)` タプル

#### `select_least_motivated_candidate(all_states)`
- **機能**: 最も意欲の低い候補者を選定（評価タスク1）
- **引数**: `all_states` - 全候補者の状態
- **戻り値**: 選定結果のテキスト

#### `rank_candidates_by_motivation(all_states)`
- **機能**: 候補者を意欲順にランキング（評価タスク2）
- **引数**: `all_states` - 全候補者の状態
- **戻り値**: ランキング結果のテキスト

#### `detect_knowledge_gaps(all_states)`
- **機能**: 知識欠損の分析と精度評価（評価タスク3）
- **引数**: `all_states` - 全候補者の状態
- **戻り値**: 分析結果と精度指標の辞書

### ヘルパー関数

#### `_analyze_interview_situation(candidate_states, asked_common_questions, current_round)`
- **機能**: 現在の面接状況を分析
- **戻り値**: 状況要約テキスト

#### `_evaluate_candidate_information_deficiency(candidate_state, company_keys)`
- **機能**: 候補者の情報欠損度を評価
- **戻り値**: `(deficiency_score, reason)` タプル

#### `_identify_deficient_candidates(candidate_states)`
- **機能**: 情報欠損が疑われる候補者を特定
- **戻り値**: 欠損候補者のリスト

#### `_should_focus_on_deficient_candidates(candidate_states, individual_question_counts)`
- **機能**: 情報欠損候補者に集中すべきか判断
- **戻り値**: `(should_focus, reason, focus_indices)` タプル

#### `_calculate_detection_metrics(llm_output_text, all_states)`
- **機能**: 知識欠損検出の精度指標を計算
- **戻り値**: 精度指標の辞書

---

## student.py - 学生クラス

学生役のLLMを扱うクラス。企業知識の管理と回答生成を担当。

### クラス: `CompanyKnowledgeManager`

#### `__init__(full_company_profile)`
- **機能**: 企業知識管理の初期化
- **引数**: `full_company_profile` (dict) - 完全な企業情報

#### `get_knowledge_for_level(level='high')`
- **機能**: 知識レベルに応じて企業情報を調整
- **引数**: `level` (str) - 'high', 'medium', 'low'
- **処理**: 必須項目とその他の項目の保持率を調整
- **戻り値**: `(knowledge_dict, coverage_str)` タプル

### クラス: `InstructionPromptManager`

#### `__init__()`
- **機能**: プロンプト管理の初期化

#### `create_instruction_prompt(preparation_level)`
- **機能**: 準備レベルに応じた指示プロンプトを生成
- **引数**: `preparation_level` (str) - 準備レベル
- **戻り値**: 指示テキスト

#### `create_prompt_string(candidate_profile, company_knowledge_tuple, conversation_history, new_question)`
- **機能**: 完全なプロンプト文字列を生成
- **引数**:
  - `candidate_profile`: 候補者プロフィール
  - `company_knowledge_tuple`: 企業知識タプル
  - `conversation_history`: 会話履歴
  - `new_question`: 新しい質問
- **戻り値**: プロンプト文字列

### クラス: `GPTApplicant`

#### `__init__(model_name)`
- **機能**: 学生役LLMの初期化
- **引数**: `model_name` (str) - 使用するモデル名

#### `generate(candidate_profile, company_knowledge_tuple, conversation_history, new_question)`
- **機能**: 学生の回答を生成
- **引数**:
  - `candidate_profile`: 候補者プロフィール
  - `company_knowledge_tuple`: 企業知識タプル
  - `conversation_history`: 会話履歴
  - `new_question`: 面接官の質問
- **戻り値**: `(response_text, token_info)` タプル

---

## model_manager.py - モデル管理

Hugging Face CLIを使ったローカルモデル管理システム。

### クラス: `HuggingFaceModelManager`

#### `__init__(cache_dir=None)`
- **機能**: モデル管理の初期化
- **引数**: `cache_dir` (str) - キャッシュディレクトリ

#### `check_hf_cli_installed()`
- **機能**: Hugging Face CLIのインストール確認
- **戻り値**: `bool` - インストール済みかどうか

#### `install_hf_cli()`
- **機能**: Hugging Face CLIのインストール
- **戻り値**: `bool` - インストール成功かどうか

#### `login_hf()`
- **機能**: Hugging Faceへのログイン
- **戻り値**: `bool` - ログイン成功かどうか

#### `get_model_info(model_key)`
- **機能**: モデル情報を取得
- **引数**: `model_key` (str) - モデルキー
- **戻り値**: モデル情報辞書

#### `list_available_models()`
- **機能**: 利用可能なモデル一覧を取得
- **戻り値**: モデルキーのリスト

#### `is_model_downloaded(model_key)`
- **機能**: モデルのダウンロード状況確認
- **引数**: `model_key` (str) - モデルキー
- **戻り値**: `bool` - ダウンロード済みかどうか

#### `download_model(model_key, force=False, progress_callback=None)`
- **機能**: モデルのダウンロード
- **引数**:
  - `model_key` (str) - モデルキー
  - `force` (bool) - 強制ダウンロード
  - `progress_callback` - 進捗コールバック関数
- **戻り値**: `bool` - ダウンロード成功かどうか

#### `get_model_path(model_key)`
- **機能**: モデルのローカルパスを取得
- **引数**: `model_key` (str) - モデルキー
- **戻り値**: モデルパス

#### `initialize_model(model_key, device="auto", quantization=True)`
- **機能**: モデルの初期化とロード
- **引数**:
  - `model_key` (str) - モデルキー
  - `device` (str) - デバイス指定
  - `quantization` (bool) - 量子化の有効/無効
- **戻り値**: `(model, tokenizer)` タプル

#### `cleanup_model(model_key)`
- **機能**: モデルの削除
- **引数**: `model_key` (str) - モデルキー
- **戻り値**: `bool` - 削除成功かどうか

#### `get_disk_usage()`
- **機能**: ディスク使用量を取得
- **戻り値**: モデル別使用量の辞書

#### `print_model_status()`
- **機能**: モデル状態を表示

---

## spreadsheet_integration.py - スプレッドシート連携

Google Apps Script (GAS) を使用してスプレッドシートに実験結果を記録する機能。

### クラス: `SpreadsheetIntegration`

#### `__init__(gas_web_app_url)`
- **機能**: スプレッドシート連携の初期化
- **引数**: `gas_web_app_url` (str) - GAS WebアプリのURL

#### `test_connection()`
- **機能**: スプレッドシートとの接続をテスト
- **戻り値**: テスト結果の辞書

#### `initialize_spreadsheet()`
- **機能**: スプレッドシートの初期化（ヘッダー行設定）
- **戻り値**: 初期化結果の辞書

#### `record_experiment_result(experiment_data)`
- **機能**: 単一の実験結果を記録
- **引数**: `experiment_data` (dict) - 実験結果データ
- **戻り値**: 記録結果の辞書

#### `record_multiple_experiment_results(experiment_data_list)`
- **機能**: 複数の実験結果を一括記録
- **引数**: `experiment_data_list` (list) - 実験結果データのリスト
- **戻り値**: 記録結果の辞書

#### `get_spreadsheet_stats()`
- **機能**: スプレッドシートの統計情報を取得
- **戻り値**: 統計情報の辞書

#### `clear_spreadsheet_data()`
- **機能**: スプレッドシートのデータをクリア
- **戻り値**: クリア結果の辞書

### ユーティリティ関数

#### `load_spreadsheet_config()`
- **機能**: スプレッドシート設定を読み込み
- **戻り値**: 設定情報の辞書

#### `create_spreadsheet_config_template()`
- **機能**: スプレッドシート設定ファイルのテンプレートを作成
- **戻り値**: `bool` - 作成成功かどうか

#### `get_spreadsheet_integration()`
- **機能**: スプレッドシート連携インスタンスを取得
- **戻り値**: `SpreadsheetIntegration`インスタンス

---

## data_generators.py - データ生成

企業情報と学生プロフィールの生成・読み込み機能。

### 関数

#### `load_company_and_candidates_from_db(set_index=None)`
- **機能**: db.jsonから企業情報と学生プロフィールを読み込み
- **引数**: `set_index` (int) - データセットインデックス（Noneの場合はランダム選択）
- **戻り値**: `(company_profile, candidate_profiles, actual_set_index)` タプル

#### `generate_company_profile()`
- **機能**: 架空の企業情報を生成
- **処理**: OpenAI APIを使用して企業情報を生成
- **戻り値**: 企業情報の辞書

#### `generate_candidate_profiles(company_profile, num_candidates)`
- **機能**: 企業情報に基づき学生プロフィールを生成
- **引数**:
  - `company_profile` (dict) - 企業情報
  - `num_candidates` (int) - 生成する候補者数
- **処理**: OpenAI APIを使用して学生プロフィールを生成
- **戻り値**: 学生プロフィールのリスト

---

## config.py - 設定ファイル

システム全体の設定を管理するファイル。

### 主要設定項目

#### API設定
- `OPENAI_API_KEY`: OpenAI APIキー
- `GENERATOR_MODEL_NAME`: データ生成用モデル
- `APPLICANT_API_MODEL`: 学生役用モデル

#### ローカルモデル設定
- `AVAILABLE_LOCAL_MODELS`: 利用可能なローカルモデルの辞書
- `LOCAL_MODEL_NAME`: デフォルトのローカルモデル

#### 面接官モデル設定
- `INTERVIEWER_MODEL_TYPE`: 面接官モデルタイプ（'local' or 'api'）
- `AVAILABLE_API_MODELS`: 利用可能なAPIモデルの辞書
- `INTERVIEWER_API_MODEL`: 面接官用APIモデル

#### 実験設定
- `NUM_CANDIDATES`: 生成する学生数
- `INTERVIEW_FLOW`: 面接フローの定義
- `USE_INTELLIGENT_DYNAMIC_FLOW`: 智的動的フローの使用フラグ
- `MAX_DYNAMIC_ROUNDS`: 動的フローでの最大ラウンド数

#### 対話設定
- `MAX_CONVERSATION_TURNS`: 最大対話回数
- `MIN_CONVERSATION_TURNS`: 最小対話回数

---

## utils.py - ユーティリティ

API呼び出しとJSON処理のユーティリティ関数。

### 関数

#### `call_openai_api(model_name, prompt)`
- **機能**: OpenAI APIを呼び出し、テキスト応答とtoken数情報を返す
- **引数**:
  - `model_name` (str) - モデル名
  - `prompt` (str) - プロンプト
- **戻り値**: `(response_text, token_info)` タプル
- **処理**:
  - APIキーの確認
  - OpenAI APIの呼び出し
  - エラーハンドリング
  - Token数情報の取得

#### `call_gemini_api(model_name, prompt)`
- **機能**: 後方互換性のため、call_openai_apiを呼び出す
- **引数**: `model_name`, `prompt`
- **戻り値**: `call_openai_api`の結果

#### `parse_json_from_response(response_text)`
- **機能**: LLMの応答からJSON部分を抽出してパース
- **引数**: `response_text` (str) - LLMの応答テキスト
- **処理**:
  - ```json```ブロックの抽出
  - JSONのパース
  - エラーハンドリング
- **戻り値**: パースされたJSONオブジェクト

---

## データ構造

### 実験結果の構造
```json
{
  "simulation_num": 1,
  "experiment_info": {
    "dataset_index": 0,
    "dataset_name": "Dataset_1",
    "interviewer_type": "api",
    "interviewer_model_name": "gpt-4o-mini",
    "interview_flow": [0, 1, 1, 1],
    "use_dynamic_flow": false,
    "total_rounds": 4,
    "timestamp": "2024-01-01T12:00:00",
    "set_index": 0
  },
  "company_profile": {...},
  "interview_transcripts": [...],
  "final_evaluations": {...},
  "accuracy_metrics": {...},
  "execution_time_seconds": 120.5
}
```

### 候補者状態の構造
```json
{
  "profile": {
    "name": "田中太郎",
    "university": "東京大学工学部",
    "gakuchika": "学生時代に力を入れたこと...",
    "interest": "AI・機械学習",
    "strength": "プログラミング",
    "preparation": "high",
    "mbti": "INTJ"
  },
  "knowledge_tuple": [
    {
      "name": "テック株式会社",
      "business": "AIソリューション開発",
      ...
    },
    "項目網羅率: 15/15 (100%)"
  ],
  "conversation_log": [
    {
      "turn": 1,
      "question": "当社についてどの程度ご存知ですか？",
      "answer": "テック株式会社は...",
      "token_info": {
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "total_tokens": 700
      }
    }
  ]
}
```

---

## エラーハンドリング

### 主要なエラー処理
1. **APIキー未設定**: 設定ファイルの確認とエラーメッセージ表示
2. **モデル初期化失敗**: フォールバック機能とエラーログ
3. **ファイル読み込みエラー**: デフォルト値の使用とエラー通知
4. **ネットワークエラー**: リトライ機能とタイムアウト処理
5. **JSON解析エラー**: エラー情報の保持とデバッグ出力

### ログ機能
- タイムスタンプ付きログメッセージ
- エラーレベルの分類
- 最大ログ件数の制限（100件）
- リアルタイムログ表示