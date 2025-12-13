# interviewer.py - 面接官役のクラス

import torch
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import call_openai_api, get_api_client
import config

# MODEL_TYPE_MAPPINGをインポート（config.pyから直接取得）
try:
    from config import MODEL_TYPE_MAPPING
except ImportError:
    # フォールバック: configモジュールから直接取得
    MODEL_TYPE_MAPPING = getattr(config, 'MODEL_TYPE_MAPPING', {})

# outlinesのインポート（オプション）
try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    print("警告: outlinesモジュールが見つかりません。構造化出力は使用できません。")

class Interviewer:
    """面接官役のLLM（ローカルモデルとAPIモデルの両方に対応）"""
    
    def __init__(self, company_profile, model_name=None, model_type='api', model=None, tokenizer=None, local_model_key=None, api_provider=None):
        """
        Args:
            company_profile (dict): 企業情報
            model_name (str, optional): APIモデル名（model_type='api'の場合）
            model_type (str): 'api' または 'local'（デフォルト: 'api'）
            model (AutoModelForCausalLM, optional): ローカルモデル（model_type='local'の場合）
            tokenizer (AutoTokenizer, optional): ローカルモデル用トークナイザ（model_type='local'の場合）
            local_model_key (str, optional): ローカルモデルのキー（例: "llama3", "ELYZA-japanese-Llama-2"）
            api_provider (str, optional): 使用するAPIプロバイダー ('openai' または 'google')
        """
        self.company = company_profile
        self.model_type = model_type
        self.model_name = model_name or config.INTERVIEWER_MODEL
        self.model = model
        self.tokenizer = tokenizer
        self.local_model_key = local_model_key
        
        # APIプロバイダーの設定（指定がない場合はconfigから取得）
        self.api_provider = api_provider or config.API_PROVIDER
        
        # モデルタイプを決定（チャットテンプレートの形式を決定するため）
        if self.model_type == 'local' and local_model_key:
            model_type_mapping = getattr(config, 'MODEL_TYPE_MAPPING', {})
            self.chat_template_type = model_type_mapping.get(local_model_key, "llama3")
        else:
            self.chat_template_type = "llama3"  # デフォルト
        
        if self.model_type == 'local' and (not self.model or not self.tokenizer):
            raise ValueError("ローカルモデルタイプには 'model' と 'tokenizer' が必要です。")
    
    def _generate_response(self, prompt, max_tokens=512):
        """モデルタイプに応じて応答を生成する"""
        if self.model_type == 'local':
            # ローカルモデルでの生成ロジック
            system_content = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
            
            # プロンプトのトークン数をチェック（Gemmaモデルの場合、コンテキストウィンドウを確認）
            if self.chat_template_type == "gemma":
                # Gemma-2-2b-jpn-itのコンテキストウィンドウは20kトークン程度
                # system_content + prompt のトークン数をチェック
                full_prompt_for_check = f"{system_content}\n\n{prompt}"
                tokenized = self.tokenizer(full_prompt_for_check, return_tensors="pt")
                input_token_count = tokenized['input_ids'].shape[1]
                max_context_tokens = 19500  # 安全マージンを考慮して19500トークンに設定（出力用に500トークン程度残す）
                
                if input_token_count > max_context_tokens:
                    print(f"警告: プロンプトのトークン数({input_token_count})が推奨制限({max_context_tokens})を超えています。")
                    print(f"生成は試みますが、コンテキストウィンドウを超える可能性があります。")
                else:
                    print(f"デバッグ: プロンプトのトークン数: {input_token_count} (制限: {max_context_tokens})")
            
            # モデルタイプに応じてメッセージ形式を変更
            if self.chat_template_type == "llama2":
                # Llama-2系モデル（ELYZA-japanese-Llama-2など）の形式
                # Llama-2系はsystemロールをサポートしていない場合があるため、userメッセージに統合
                messages = [
                    {"role": "user", "content": f"{system_content}\n\n{prompt}"}
                ]
                # Llama-2系ではadd_generation_promptをFalseにする場合がある
                try:
                    encoded = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                    )
                    # BatchEncodingまたはTensorのどちらかが返される可能性がある
                    if isinstance(encoded, dict) and 'input_ids' in encoded:
                        inputs = encoded['input_ids'].to(self.model.device)
                    elif hasattr(encoded, 'input_ids'):
                        inputs = encoded.input_ids.to(self.model.device)
                    else:
                        # Tensorが直接返された場合
                        inputs = encoded.to(self.model.device)
                except Exception as e:
                    # フォールバック: 直接プロンプトをエンコード
                    full_prompt = f"{system_content}\n\n{prompt}"
                    encoded = self.tokenizer(full_prompt, return_tensors="pt")
                    inputs = encoded['input_ids'].to(self.model.device)
            elif self.chat_template_type == "gemma":
                # Gemmaモデル（gemma-2-2b-jpn-itなど）の形式
                # Gemmaモデルはsystemロールをサポートしていないため、userメッセージに統合し、apply_chat_templateを使わない
                # Gemmaモデル用のプロンプト形式: <start_of_turn>user\n{プロンプト}<end_of_turn>\n<start_of_turn>model\n
                full_prompt = f"<start_of_turn>user\n{system_content}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                encoded = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
                inputs = encoded['input_ids'].to(self.model.device)
            elif self.chat_template_type == "qwen":
                # Qwen系モデル（Qwen3-4B-Instructなど）の形式
                # Qwen系はsystemロールをサポートしているが、形式が異なる場合がある
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
                try:
                    encoded = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                    )
                    # BatchEncodingまたはTensorのどちらかが返される可能性がある
                    if isinstance(encoded, dict) and 'input_ids' in encoded:
                        inputs = encoded['input_ids'].to(self.model.device)
                    elif hasattr(encoded, 'input_ids'):
                        inputs = encoded.input_ids.to(self.model.device)
                    else:
                        # Tensorが直接返された場合
                        inputs = encoded.to(self.model.device)
                except Exception as e:
                    # フォールバック: 直接プロンプトをエンコード
                    full_prompt = f"{system_content}\n\n{prompt}"
                    encoded = self.tokenizer(full_prompt, return_tensors="pt")
                    inputs = encoded['input_ids'].to(self.model.device)
            elif self.chat_template_type == "llm-jp":
                # llm-jpモデル用の形式（手動構築）
                # Instruction形式: ### 指示:\n{system}\n{user}\n\n### 応答:\n
                # 末尾の「質問:」などはモデルを混乱させる可能性があるため削除
                clean_prompt = prompt.strip()
                if clean_prompt.endswith("質問:"):
                    clean_prompt = clean_prompt[:-3].strip()
                
                # system_contentとpromptを結合
                combined_input = f"{system_content}\n\n{clean_prompt}"
                
                full_prompt = f"以下は、タスクを説明する指示です。指示を適切に完了する応答を記述してください。\n### 指示:\n{combined_input}\n### 応答:\n"
                encoded = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
                if hasattr(encoded, "input_ids"):
                    inputs = encoded.input_ids.to(self.model.device)
                elif isinstance(encoded, dict) and "input_ids" in encoded:
                    inputs = encoded["input_ids"].to(self.model.device)
                else:
                    inputs = encoded.to(self.model.device)
            else:
                # Llama-3系やその他のモデル（デフォルト）
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
                
                encoded = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                )
                # BatchEncodingまたはTensorのどちらかが返される可能性がある
                if isinstance(encoded, dict) and 'input_ids' in encoded:
                    inputs = encoded['input_ids'].to(self.model.device)
                elif hasattr(encoded, 'input_ids'):
                    inputs = encoded.input_ids.to(self.model.device)
                else:
                    # Tensorが直接返された場合
                    inputs = encoded.to(self.model.device)
            
            attention_mask = torch.ones_like(inputs).to(self.model.device)
            
            # Gemmaモデルの場合、生成パラメータを調整
            if self.chat_template_type == "gemma":
                # Gemmaモデル用の生成パラメータ（より高いtemperatureとtop_pで多様な応答を促す）
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": True,
                    "temperature": 0.8,  # より高いtemperature
                    "top_p": 0.95,  # より高いtop_p
                    "repetition_penalty": 1.1,  # より低いrepetition_penalty
                    "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                }
            elif self.chat_template_type == "llm-jp":
                # llm-jp-1.8bは小さいモデルなので、制約を緩める
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "repetition_penalty": 1.05, # 強くしすぎるとおかしくなることがある
                    "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                }
            else:
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2
                }
            
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                **generation_kwargs
            )
            
            response = outputs[0][inputs.shape[-1]:]
            decoded_response = self.tokenizer.decode(response, skip_special_tokens=True).strip()
            
            # デバッグ: 空の応答をチェック
            if not decoded_response and self.chat_template_type == "gemma":
                print(f"警告: Gemmaモデルの応答が空です。入力長: {inputs.shape[-1]}, 出力長: {len(response)}")
                # デバッグ: 生のトークンIDを確認
                if len(response) > 0:
                    print(f"デバッグ: 生のトークンID: {response.tolist()[:10]}")
                    # 特殊トークンを除外せずにデコードしてみる
                    decoded_with_special = self.tokenizer.decode(response, skip_special_tokens=False)
                    print(f"デバッグ: 特殊トークン込みのデコード結果: {decoded_with_special[:200]}")
            
            return decoded_response, None
        
        elif self.model_type == 'api':
            # APIモデルでの生成ロジック
            system_prompt = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
            full_prompt = f"システム指示: {system_prompt}\n\nユーザー指示:\n{prompt}"
            response_text, token_info = call_openai_api(self.model_name, full_prompt, provider=self.api_provider)
            return response_text, token_info
        
        else:
            raise ValueError(f"無効なモデルタイプです: {self.model_type}")
    
    def ask_common_question(self, asked_questions):
        """全候補者に対する全体質問を生成"""
        all_company_keys = list(self.company.keys())
        history_str = "\n".join(f"- {q}" for q in asked_questions) if asked_questions else "  (なし)"

        prompt = f"""あなたは、面接全体を俯瞰し、全候補者の理解度を効率的に測る、戦略的な採用面接官です。
これから全候補者に対して同じ共通質問をします。

# あなたが質問できる企業情報の項目リスト
{all_company_keys}

# これまでに行った全ての質問（重複しないように）
{history_str}

【全体質問の戦略的役割】
全体質問は以下の目的で使用されます：
1. **比較基準の確立**: 全候補者が同じ条件で回答するため、公平な比較が可能
2. **基盤情報の収集**: 候補者間の差別化に必要な基本的な企業理解度を測定
3. **効率的な情報収集**: 一度の質問で全候補者から情報を収集し、時間を節約
4. **共通トピックの深掘り**: 特定の重要な企業情報について全員の見解を比較

【全体質問を選ぶべき戦略的状況】
- 候補者間の比較材料が不足している場合
- 特定の重要な企業情報について全員の理解度を測りたい場合
- 個別質問で深掘りする前に基盤となる情報が必要な場合
- 効率的に情報収集を進めたい場合

# 指示
1.  **カバレッジ分析**: まだ質問されていない「項目リスト」の中の項目を探してください。**過去の質問と重複するトピックは絶対に避けてください**。
2.  **ターゲット選定**: **完全に新しいトピック**を最優先で選んでください。前の質問の関連質問ではなく、全く異なる分野（例：前の質問が「ビジョン」なら、次は「福利厚生」や「技術スタック」など）に切り替えることを意識してください。
3.  **戦略的質問生成**: 選んだ新しいトピックについて、候補者の知識を試すための質問を作成してください。
4.  **比較可能性の確保**: 全候補者が回答できる形式であることを確認してください。

思考プロセスや前置きは一切含めず、質問文だけを出力してください。
質問:"""
        
        question, token_info = self._generate_response(prompt, max_tokens=512)
        return question.strip(), token_info
    
    def ask_question(self, conversation_history, asked_questions=None):
        """特定の候補者への個別質問を生成"""
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history])
        all_company_keys = list(self.company.keys())

        prompt = f"""あなたは、学生の企業研究の深さを測る、戦略的な採用面接官です。
# あなたが質問できる企業情報の項目リスト
{all_company_keys}
# これまでの会話履歴
{history_str if history_str else "（まだ会話はありません）"}

【個別質問の戦略的役割】
個別質問は以下の目的で使用されます：
1. **深掘り調査**: 特定の候補者の回答をより深く探り、詳細な情報を収集
2. **個人的動機の探求**: 候補者固有の志望動機や背景を理解
3. **知識欠損の特定**: この候補者が特に不足している企業知識を特定
4. **差別化要因の発見**: 他の候補者との違いを明確にする情報を収集
5. **曖昧な回答の明確化**: 以前の回答で不明確だった部分を明確にする

【個別質問を選ぶべき戦略的状況】
- 特定の候補者の回答をより深く探る必要がある場合
- 候補者ごとに異なる角度からの質問が効果的な場合
- 志望度の判定に必要な個人的な動機を探る必要がある場合
- 十分な全体質問が既に実施されている場合
- 情報が不足している候補者が特定されている場合

# 指示
1.  **カバレッジ分析**: 上記の「項目リスト」と「会話履歴」を比較し、**まだ一度も話題に上がっていない項目**を特定してください。
2.  **ターゲット選定**: **未言及の項目を最優先**して質問対象に選んでください。すでに話題に出た項目に関する深掘りは、その項目について明らかに理解が浅い場合のみに限定してください。
3.  **戦略的質問生成**: 選んだ項目について、学生が本当に理解しているかを確認するための鋭い質問を作成してください。「〜についてどう思いますか？」のような抽象的な質問ではなく、具体的な事実や見解を問う質問にしてください。
4.  **知識欠損のあぶり出し**: ターゲットとした項目について、知識がなければ答えられないような質問を投げかけ、知識の有無を判定しやすくしてください。

思考プロセスや前置きは一切含めず、質問文だけを出力してください。
質問:"""
        
        question, token_info = self._generate_response(prompt, max_tokens=512)
        return question.strip(), token_info
    
    def select_least_motivated_candidate(self, candidate_states, current_round=1, total_rounds=10):
        """最も志望度が低いと思われる候補者を選定（構造化出力を使用）"""
        
        # 各候補者の会話ログを整形
        candidates_info = []
        candidate_names = []
        for i, state in enumerate(candidate_states):
            name = state['profile']['name']
            candidate_names.append(name)
            conversations = []
            for turn in state['conversation_log']:
                conversations.append(f"Q: {turn['question']}\nA: {turn['answer']}")
            conv_str = "\n".join(conversations)
            candidates_info.append(f"候補者{i+1}: {name}\n{conv_str}")
        
        all_conversations = "\n\n".join(candidates_info)
        
        # Pydanticモデルで選定結果の構造を定義
        class LeastMotivatedResult(BaseModel):
            candidate_name: str = Field(description="最も志望度が低い候補者名。判断できない場合は'none'または有力な候補を記入")
            confidence: int = Field(description="評価の確信度 (1: 全く自信なし/判断不能, 5: 非常に強い確信)", ge=1, le=5)
            reason: str = Field(description="その判断に至った理由")
        
        prompt = f"""以下の候補者の面接内容を分析し、最も志望度が低いと思われる候補者を選んでください。

# 現在の状況
- 現在のラウンド: {current_round} / {total_rounds}

# 面接内容
{all_conversations}

# 候補者名リスト
{', '.join(candidate_names)}

# 分析のポイント
- 企業知識の深さ
- 回答の具体性
- 熱意の表現

# 確信度のガイドライン
現在はラウンド {current_round} です。以下の基準で確信度を決定してください：
- 序盤 (1-{int(total_rounds * 0.3)}): まだ情報が少ない段階です。決定的な証拠がない限り、確信度は1-2に留めてください。「様子見」が適切な判断です。
- 中盤 ({int(total_rounds * 0.3) + 1}-{int(total_rounds * 0.7)}): 仮説を立てる段階です。確信度は3前後を目安にしてください。
- 終盤 ({int(total_rounds * 0.7) + 1}-{total_rounds}): 結論を出す段階です。これまでの情報を総合し、4-5の確信度で判断してください。

指示: 最も志望度が低いと判断される候補者名と、その確信度（1-5）を出力してください。情報不足の場合は低い確信度を出力してください。
"""
        
        # ローカルモデルの場合、outlinesを使用
        if self.model_type == 'local' and OUTLINES_AVAILABLE:
            try:
                # outlinesで構造化生成
                # outlines.models.transformersはモジュールなので、正しいクラスを使用
                try:
                    from outlines.models import transformers as outlines_transformers
                    outlines_model = outlines_transformers.Transformers(self.model, self.tokenizer)
                except (AttributeError, ImportError, TypeError):
                    # フォールバック: 直接インポートを試みる
                    import outlines.models.transformers as outlines_transformers
                    outlines_model = outlines_transformers(self.model, self.tokenizer)
                # outlinesのバージョンによってAPIが異なる可能性があるため、複数の方法を試す
                try:
                    # 新しいAPI（outlines 0.0.40以降）
                    generator = outlines.generate.json(outlines_model, LeastMotivatedResult)
                except AttributeError:
                    try:
                        # 代替API
                        from outlines import generate
                        generator = generate.json(outlines_model, LeastMotivatedResult)
                    except (AttributeError, ImportError):
                        # フォールバック: outlinesを使わずに通常の生成を使用
                        raise AttributeError("outlines.generate is not available")
                
                messages = [
                    {"role": "system", "content": "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"},
                    {"role": "user", "content": prompt}
                ]
                
                # チャットテンプレートを適用してプロンプトを作成
                # モデルタイプに応じてチャットテンプレートの処理を調整
                if self.chat_template_type == "llama2":
                    # Llama-2系はsystemロールをサポートしていないため、userメッセージに統合
                    user_message = f"{messages[0]['content']}\n\n{messages[1]['content']}"
                    formatted_prompt = user_message
                elif self.chat_template_type == "gemma":
                    # Gemma系はsystemロールをサポートしていないため、userメッセージに統合
                    user_message = f"{messages[0]['content']}\n\n{messages[1]['content']}"
                    formatted_prompt = user_message
                elif self.chat_template_type == "qwen":
                    # Qwen系は通常のチャットテンプレートを使用
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                elif self.chat_template_type == "llm-jp":
                    # llm-jp系（手動構築）
                    # 公式フォーマット: 以下は、タスクを説明する指示です。指示を適切に完了する応答を記述してください。\n### 指示:\n{input}\n### 応答:\n
                    user_message = f"以下は、タスクを説明する指示です。指示を適切に完了する応答を記述してください。\n### 指示:\n{messages[0]['content']}\n\n{messages[1]['content']}\n### 応答:\n"
                    formatted_prompt = user_message
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                
                # JSON形式で生成
                result = generator(formatted_prompt, max_tokens=256)
                
                # 結果を文字列形式に変換
                least_motivated_result = LeastMotivatedResult.model_validate(result)
                
                return least_motivated_result, None
                
            except Exception as e:
                print(f"警告: outlinesでの構造化生成に失敗しました: {e}")
                print("フォールバック: 通常の生成方法を使用します。")
                # フォールバック: 通常の生成方法
                try:
                    evaluation, token_info = self._generate_response(prompt, max_tokens=256)
                    # フォールバック時は候補者名を抽出を試みる
                    extracted_name = self._extract_candidate_name_from_text(evaluation, candidate_names)
                    # 簡易的なResultオブジェクトを作成
                    class FallbackResult:
                        def __init__(self, name, conf, reason):
                            self.candidate_name = name
                            self.confidence = conf
                            self.reason = reason
                    
                    return FallbackResult(extracted_name if extracted_name else "none", 1, evaluation), token_info
                except Exception as fallback_error:
                    print(f"警告: フォールバック生成も失敗しました: {fallback_error}")
                    class FallbackResult:
                        def __init__(self, name, conf, reason):
                            self.candidate_name = name
                            self.confidence = conf
                            self.reason = reason
                    # 最後の手段: 最初の候補者を返す
                    return FallbackResult(candidate_names[0] if candidate_names else "none", 1, "Error"), None
        
        # APIモデルの場合、JSONモードを使用
        elif self.model_type == 'api':
            try:
                # OpenAI APIのJSONモードを使用
                system_prompt = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
                
                # OpenAI APIのJSONモードで呼び出し
                client = get_api_client(self.api_provider)
                
                # 特定のモデル（gpt-5-miniなど）はtemperatureとmax_tokensパラメータをサポートしていない
                temperature_unsupported_models = ["gpt-5-mini", "gpt-5"]
                request_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt + "\n\n出力形式: JSON形式で、以下の構造で出力してください:\n{\n  \"candidate_name\": \"候補者名\",\n  \"confidence\": 1,\n  \"reason\": \"理由\"\n}"}
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                # temperatureとmax_tokensをサポートしているモデルのみに設定
                if self.model_name not in temperature_unsupported_models:
                    request_params["temperature"] = 0.6
                    request_params["max_tokens"] = 256
                else:
                    request_params["max_completion_tokens"] = 256
                
                response = client.chat.completions.create(**request_params)
                
                result_json = json.loads(response.choices[0].message.content)
                least_motivated_result = LeastMotivatedResult.model_validate(result_json)
                
                token_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                
                return least_motivated_result, token_info
                
            except Exception as e:
                print(f"警告: JSONモードでの生成に失敗しました: {e}")
                print("フォールバック: 通常の生成方法を使用します。")
                # フォールバック: 通常の生成方法
                evaluation, token_info = self._generate_response(prompt, max_tokens=256)
                # フォールバック時は候補者名を抽出を試みる
                extracted_name = self._extract_candidate_name_from_text(evaluation, candidate_names)
                
                class FallbackResult:
                    def __init__(self, name, conf, reason):
                        self.candidate_name = name
                        self.confidence = conf
                        self.reason = reason
                        
                return FallbackResult(extracted_name if extracted_name else "none", 1, evaluation), token_info
        
        # フォールバック: 通常の生成方法
        else:
            evaluation, token_info = self._generate_response(prompt, max_tokens=256)
            # フォールバック時は候補者名を抽出を試みる
            extracted_name = self._extract_candidate_name_from_text(evaluation, candidate_names)
            
            class FallbackResult:
                def __init__(self, name, conf, reason):
                    self.candidate_name = name
                    self.confidence = conf
                    self.reason = reason
                    
            return FallbackResult(extracted_name if extracted_name else "none", 1, evaluation), token_info
    
    def _extract_candidate_name_from_text(self, text, candidate_names):
        """テキストから候補者名を抽出する（フォールバック用）"""
        import re
        # テキスト内の「學生」を「学生」に統一
        normalized_text = text.replace('學生', '学生')
        
        # 正規化された候補者名リスト
        normalized_names = {re.sub(r'[\s()（）、,，。*]', '', name): name for name in candidate_names}
        
        # テキストから候補者名を探す
        for norm_name, orig_name in normalized_names.items():
            if norm_name in re.sub(r'[\s()（）、,，。*]', '', normalized_text):
                return orig_name
        
        # 部分一致で探す（「学生」+ 英数字のパターン）
        for orig_name in candidate_names:
            # 「学生」+ 英数字のパターンを抽出
            pattern = orig_name.replace('学生', r'(?:学生|學生)')
            if re.search(pattern, normalized_text):
                return orig_name
        
        # さらに柔軟なマッチング: 英数字部分のみでマッチ
        for orig_name in candidate_names:
            # 英数字部分を抽出（例: "学生C2" -> "C2"）
            match = re.search(r'([A-Z]{1,3}\d{0,2})', orig_name)
            if match:
                code = match.group(1)
                # テキスト内にこのコードが含まれているか
                if code in normalized_text:
                    return orig_name
        
        return None
    
    def rank_candidates_by_motivation(self, candidate_states):
        """候補者を志望度順にランキング（構造化出力を使用）"""
        
        # 各候補者の会話ログを整形
        candidates_info = []
        candidate_names = []
        for i, state in enumerate(candidate_states):
            name = state['profile']['name']
            candidate_names.append(name)
            conversations = []
            for turn in state['conversation_log']:
                conversations.append(f"Q: {turn['question']}\nA: {turn['answer']}")
            conv_str = "\n".join(conversations)
            candidates_info.append(f"候補者{i+1}: {name}\n{conv_str}")
        
        all_conversations = "\n\n".join(candidates_info)
        
        # Pydanticモデルでランキング構造を定義（シンプル版）
        class RankingEntry(BaseModel):
            rank: int = Field(description="順位（1位が最も志望度が低い）")
            candidate_name: str = Field(description="候補者名")
        
        class RankingResult(BaseModel):
            ranking: List[RankingEntry] = Field(description="志望度が低い順のランキング")
        
        prompt = f"""以下の候補者の面接内容を分析し、志望度の低い順にランキングしてください。

# 面接内容
{all_conversations}

# 候補者名リスト
{', '.join(candidate_names)}

# ランキングの基準
- 企業知識の深さ
- 回答の具体性
- 熱意の表現

指示: 志望度が低い順（1位が最も志望度が低い）に、候補者名のみを簡潔にランキングしてください。
"""
        
        # ローカルモデルの場合、outlinesを使用
        if self.model_type == 'local' and OUTLINES_AVAILABLE:
            try:
                # outlinesで構造化生成
                # outlines.models.transformersでモデルをラップ
                try:
                    from outlines.models import transformers as outlines_transformers
                    outlines_model = outlines_transformers.Transformers(self.model, self.tokenizer)
                except (AttributeError, ImportError, TypeError):
                    # フォールバック: 直接インポートを試みる
                    import outlines.models.transformers as outlines_transformers
                    outlines_model = outlines_transformers(self.model, self.tokenizer)
                
                # outlinesのバージョンによってAPIが異なる可能性があるため、複数の方法を試す
                try:
                    # 新しいAPI（outlines 0.0.40以降）
                    generator = outlines.generate.json(outlines_model, RankingResult)
                except AttributeError:
                    try:
                        # 代替API
                        from outlines import generate
                        generator = generate.json(outlines_model, RankingResult)
                    except (AttributeError, ImportError):
                        # フォールバック: outlinesを使わずに通常の生成を使用
                        raise AttributeError("outlines.generate is not available")
                
                messages = [
                    {"role": "system", "content": "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"},
                    {"role": "user", "content": prompt}
                ]
                
                # チャットテンプレートを適用してプロンプトを作成
                # モデルタイプに応じてチャットテンプレートの処理を調整
                if self.chat_template_type == "llama2":
                    # Llama-2系はsystemロールをサポートしていないため、userメッセージに統合
                    user_message = f"{messages[0]['content']}\n\n{messages[1]['content']}"
                    formatted_prompt = user_message
                elif self.chat_template_type == "gemma":
                    # Gemma系はsystemロールをサポートしていないため、userメッセージに統合
                    user_message = f"{messages[0]['content']}\n\n{messages[1]['content']}"
                    formatted_prompt = user_message
                elif self.chat_template_type == "qwen":
                    # Qwen系は通常のチャットテンプレートを使用
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                elif self.chat_template_type == "llm-jp":
                    # llm-jp系（手動構築）
                    # 公式フォーマット: 以下は、タスクを説明する指示です。指示を適切に完了する応答を記述してください。\n### 指示:\n{input}\n### 応答:\n
                    user_message = f"以下は、タスクを説明する指示です。指示を適切に完了する応答を記述してください。\n### 指示:\n{messages[0]['content']}\n\n{messages[1]['content']}\n### 応答:\n"
                    formatted_prompt = user_message
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                
                # JSON形式で生成
                result = generator(formatted_prompt, max_tokens=512)
                
                # 結果を文字列形式に変換
                ranking_result = RankingResult.model_validate(result)
                
                # シンプルな形式に変換
                ranking_text = ""
                for entry in ranking_result.ranking:
                    ranking_text += f"{entry.rank}位: {entry.candidate_name}\n"
                
                return ranking_text.strip(), None
                
            except Exception as e:
                print(f"警告: outlinesでの構造化生成に失敗しました: {e}")
                print("フォールバック: 通常の生成方法を使用します。")
                # フォールバック: 通常の生成方法
                ranking, token_info = self._generate_response(prompt, max_tokens=512)
                return ranking, token_info
        
        # APIモデルの場合、JSONモードを使用
        elif self.model_type == 'api':
            try:
                # OpenAI APIのJSONモードを使用
                system_prompt = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
                
                # OpenAI APIのJSONモードで呼び出し
                client = get_api_client(self.api_provider)
                
                # 特定のモデル（gpt-5-miniなど）はtemperatureとmax_tokensパラメータをサポートしていない
                temperature_unsupported_models = ["gpt-5-mini", "gpt-5"]
                request_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt + "\n\n出力形式: JSON形式で、以下の構造で出力してください:\n{\n  \"ranking\": [\n    {\"rank\": 1, \"candidate_name\": \"候補者名\"},\n    {\"rank\": 2, \"candidate_name\": \"候補者名\"},\n    {\"rank\": 3, \"candidate_name\": \"候補者名\"}\n  ]\n}"}
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                # temperatureとmax_tokensをサポートしているモデルのみに設定
                if self.model_name not in temperature_unsupported_models:
                    request_params["temperature"] = 0.6
                    request_params["max_tokens"] = 512
                else:
                    request_params["max_completion_tokens"] = 512
                
                response = client.chat.completions.create(**request_params)
                
                result_json = json.loads(response.choices[0].message.content)
                ranking_result = RankingResult.model_validate(result_json)
                
                # シンプルな形式に変換
                ranking_text = ""
                for entry in ranking_result.ranking:
                    ranking_text += f"{entry.rank}位: {entry.candidate_name}\n"
                
                token_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                
                return ranking_text.strip(), token_info
                
            except Exception as e:
                print(f"警告: JSONモードでの生成に失敗しました: {e}")
                print("フォールバック: 通常の生成方法を使用します。")
                # フォールバック: 通常の生成方法
                ranking, token_info = self._generate_response(prompt, max_tokens=512)
                return ranking, token_info
        
        # フォールバック: 通常の生成方法
        else:
            ranking, token_info = self._generate_response(prompt, max_tokens=512)
            return ranking, token_info
    
    def _format_all_conversations(self, all_states, max_rounds=None):
        """全候補者の会話ログを整形するヘルパー
        
        Args:
            all_states: 候補者の状態リスト
            max_rounds: 最新のNラウンドのみを含める（Noneの場合は全て）
        """
        full_log = ""
        for i, state in enumerate(all_states):
            profile = state['profile']
            conversation_log = state['conversation_log']
            
            # max_roundsが指定されている場合は、最新のNラウンドのみを使用
            if max_rounds is not None and len(conversation_log) > max_rounds:
                conversation_log = conversation_log[-max_rounds:]
                full_log += f"--- 候補者{i+1}: {profile.get('name')} (最新{max_rounds}ラウンドのみ) ---\n"
            else:
                full_log += f"--- 候補者{i+1}: {profile.get('name')} ---\n"
            
            history_str = "\n".join([f"  面接官: {turn['question']}\n  {profile.get('name')}: {turn['answer']}" for turn in conversation_log])
            full_log += f"会話履歴:\n{history_str}\n\n"
        return full_log.strip()
    
    def _calculate_detection_metrics(self, llm_output_text, all_states):
        """LLMの出力と正解データを比較し、TP/FP/FNなどの性能メトリクスを計算する"""
        print(f"\n[デバッグ] LLM出力テキスト（最初の1000文字）:\n{llm_output_text[:1000]}\n")
        
        evaluation_results = {}
        candidate_states_map = {s['profile']['name']: s for s in all_states}
        
        # より柔軟なセクション分割（"- 候補者名:" または "候補者名:" のパターンに対応）
        # まず "- " で始まる行で分割（改行の前後を考慮）
        sections = re.split(r'(?=\n?-\s+[^\n]+:)', llm_output_text, re.MULTILINE)
        
        # もし分割できなかったら、候補者名のパターンで分割
        if len(sections) <= 1:
            # 候補者名のパターンで分割（"学生" または "student" で始まる、大文字小文字を区別しない）
            # "学生KKK1:" や "studentKKK1：" のパターンに対応
            sections = re.split(r'(?=\n?(?:学生|student)[^\n:]+[：:])', llm_output_text, re.MULTILINE | re.IGNORECASE)
        
        # さらに、より柔軟なパターンで分割を試す
        if len(sections) <= 1:
            # "- 候補者名:" または "候補者名:" で始まる行（改行の前後を考慮）
            sections = re.split(r'(?=\n?-\s*[^\n]+[：:])', llm_output_text, re.MULTILINE)
        
        # 最初のセクションが空の場合は削除
        if sections and not sections[0].strip():
            sections = sections[1:]
        
        # セクションがまだ1つしかない場合、手動で分割を試す
        if len(sections) <= 1:
            # "- 学生" で始まる部分を手動で分割
            parts = re.finditer(r'-\s*([^\n]+?):', llm_output_text)
            section_starts = []
            for match in parts:
                section_starts.append(match.start())
            
            if len(section_starts) > 1:
                sections = []
                for i in range(len(section_starts)):
                    start = section_starts[i]
                    end = section_starts[i + 1] if i + 1 < len(section_starts) else len(llm_output_text)
                    sections.append(llm_output_text[start:end])
        
        print(f"[デバッグ] セクション数: {len(sections)}")
        print(f"[デバッグ] 候補者名マップ: {list(candidate_states_map.keys())}\n")
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # 候補者名の抽出をより柔軟に
            candidate_name_raw = None
            
            # パターン1: "- 候補者名:" または "候補者名:" で始まる行
            first_line = section.split('\n', 1)[0] if '\n' in section else section
            name_match = re.search(r'[-*]*\s*([^\n:]+?)[：:]', first_line)
            if name_match:
                candidate_name_raw = name_match.group(1).strip()
                # マークダウン記号を除去
                candidate_name_raw = re.sub(r'[*#\s]+', '', candidate_name_raw)
            
            # パターン2: 候補者名パターンを直接検索
            if not candidate_name_raw:
                name_pattern = r'(学生[A-Z]{1,3}\d{0,2})'
                name_match = re.search(name_pattern, first_line)
                if name_match:
                    candidate_name_raw = name_match.group(1)
            
            if not candidate_name_raw:
                print(f"[デバッグ] 警告: 候補者名を抽出できませんでした。セクションの最初の行: {first_line[:100]}")
                continue
            
            print(f"[デバッグ] 抽出された候補者名（生）: '{candidate_name_raw}'")
            
            # 候補者名のマッチング（部分一致も試す）
            candidate_name = None
            # 完全一致を試す
            if candidate_name_raw in candidate_states_map:
                candidate_name = candidate_name_raw
            else:
                # 部分一致を試す（例: "studentKKK1" → "学生KKK1"）
                for mapped_name in candidate_states_map.keys():
                    # 数字部分を抽出して比較
                    raw_numbers = ''.join(re.findall(r'\d+', candidate_name_raw))
                    mapped_numbers = ''.join(re.findall(r'\d+', mapped_name))
                    if raw_numbers == mapped_numbers and raw_numbers:
                        candidate_name = mapped_name
                        print(f"[デバッグ] 数字部分でマッチ: '{candidate_name_raw}' → '{candidate_name}'")
                        break
                    # 名前の一部が含まれているかチェック
                    if candidate_name_raw in mapped_name or mapped_name in candidate_name_raw:
                        candidate_name = mapped_name
                        print(f"[デバッグ] 部分文字列でマッチ: '{candidate_name_raw}' → '{candidate_name}'")
                        break
                    # "学生" と "student" の変換を試す
                    if 'student' in candidate_name_raw.lower() and '学生' in mapped_name:
                        raw_suffix = candidate_name_raw.lower().replace('student', '').strip()
                        mapped_suffix = mapped_name.replace('学生', '').strip()
                        if raw_suffix == mapped_suffix:
                            candidate_name = mapped_name
                            print(f"[デバッグ] 英語/日本語変換でマッチ: '{candidate_name_raw}' → '{candidate_name}'")
                            break
            
            if candidate_name is None:
                print(f"[デバッグ] 警告: 候補者名 '{candidate_name_raw}' が候補者リストに存在しません")
                print(f"[デバッグ] 利用可能な候補者名: {list(candidate_states_map.keys())}")
                continue
            
            print(f"[デバッグ] 最終的に使用する候補者名: '{candidate_name}'")
            
            state = candidate_states_map[candidate_name]
            note = None
            detected_missing_keys = set()
            
            print(f"[デバッグ] セクション内容（最初の200文字）:\n{section[:200]}\n")
            
            # より柔軟なパターンでキー行を検索
            key_line_match = re.search(r"欠損項目キー:\s*(\[.*?\])", section, re.DOTALL)
            if not key_line_match:
                # 別のパターン: 改行を含む可能性がある
                key_line_match = re.search(r"欠損項目キー:\s*\[(.*?)\]", section, re.DOTALL)
            
            if key_line_match:
                try:
                    if key_line_match.lastindex >= 1:
                        keys_content = key_line_match.group(1)
                        # JSON配列形式に変換を試みる
                        keys_str = f"[{keys_content}]"
                        print(f"[デバッグ] 抽出されたキー文字列: {keys_str}")
                        # まずJSONとしてパースを試みる
                        try:
                            parsed = json.loads(keys_str)
                            # パース結果がリストの場合、フラット化を試みる
                            if isinstance(parsed, list):
                                # 二重リストの場合を処理
                                flat_list = []
                                for item in parsed:
                                    if isinstance(item, list):
                                        flat_list.extend(item)
                                    elif isinstance(item, str):
                                        flat_list.append(item)
                                detected_missing_keys = set(flat_list)
                            elif isinstance(parsed, str):
                                detected_missing_keys = {parsed}
                            else:
                                detected_missing_keys = set(parsed) if hasattr(parsed, '__iter__') else {str(parsed)}
                        except json.JSONDecodeError:
                            # JSONパースに失敗した場合、手動で抽出
                            # クォートで囲まれた文字列を抽出
                            key_matches = re.findall(r'["\']([^"\']+)["\']', keys_content)
                            if key_matches:
                                detected_missing_keys = set(key_matches)
                            else:
                                # クォートがない場合、カンマ区切りで抽出
                                keys_list = [k.strip() for k in keys_content.split(',') if k.strip()]
                                detected_missing_keys = set(keys_list)
                        print(f"[デバッグ] パース成功: 検出された欠損キー = {detected_missing_keys}")
                except Exception as e:
                    note = f"Detected '欠損項目キー' but failed to parse: {e}"
                    print(f"[デバッグ] パースエラー: {e}")
                    import traceback
                    print(f"[デバッグ] トレースバック: {traceback.format_exc()}")
            else:
                note = "Candidate block found, but '欠損項目キー' line is missing."
                print(f"[デバッグ] 警告: '欠損項目キー' の行が見つかりません")

            # mochiではknowledgeが辞書形式
            possessed_knowledge = state['knowledge']
            actual_missing_keys = {key for key, value in possessed_knowledge.items() if not value}
            actual_possessed_keys = {key for key, value in possessed_knowledge.items() if value}
            all_company_keys = set(list(self.company.keys()))
            detect_possessed_keys = all_company_keys.difference(detected_missing_keys)
            
            print(f"[デバッグ] {candidate_name}:")
            print(f"  実際の欠損キー: {actual_missing_keys}")
            print(f"  実際の保持キー: {actual_possessed_keys}")
            print(f"  検出された欠損キー: {detected_missing_keys}")
            print(f"  全企業キー: {all_company_keys}\n")

            true_positives = actual_missing_keys.intersection(detected_missing_keys)
            true_negatives = actual_possessed_keys.intersection(detect_possessed_keys)
            false_positives = detected_missing_keys.difference(actual_missing_keys)
            false_negatives = actual_missing_keys.difference(detected_missing_keys)
            tp_count, tn_count, fp_count, fn_count = len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp_count + tn_count) / len(all_company_keys) if all_company_keys else 0.0

            result = {
                "metrics": {
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "accuracy": round(accuracy, 3),
                    "f1_score": round(f1_score, 3),
                    "true_positives": tp_count,
                    "true_negatives": tn_count,
                    "false_positives": fp_count,
                    "false_negatives": fn_count,
                },
                "details": {
                    "correctly_detected_gaps (TP)": list(true_positives),
                    "correctly_detected_knowns (TN)": list(true_negatives),
                    "incorrectly_detected_gaps (FP)": list(false_positives),
                    "missed_gaps (FN)": list(false_negatives),
                }
            }
            if note:
                result["note"] = note
            evaluation_results[candidate_name] = result
            
            print(f"[デバッグ] {candidate_name} のメトリクス:")
            print(f"  TP: {tp_count}, TN: {tn_count}, FP: {fp_count}, FN: {fn_count}")
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}, F1: {f1_score:.3f}\n")

        for state in all_states:
            if state['profile']['name'] not in evaluation_results:
                actual_missing_keys = {key for key, value in state['knowledge'].items() if not value}
                evaluation_results[state['profile']['name']] = {
                    "metrics": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "accuracy": 0.0,
                        "f1_score": 0.0,
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": len(actual_missing_keys)
                    },
                    "details": {
                        "correctly_detected_gaps (TP)": [],
                        "incorrectly_detected_gaps (FP)": [],
                        "missed_gaps (FN)": list(actual_missing_keys)
                    },
                    "note": "LLM output for this candidate was not found or failed to parse."
                }
        return evaluation_results
    
    def detect_knowledge_gaps(self, all_states, least_motivated_eval, ranking_eval, current_round=1, total_rounds=10):
        """評価タスク3: 知識欠損の定性分析と定量評価を同時に行う"""
        
        full_company_info_str = json.dumps(self.company, ensure_ascii=False, indent=2)
        
        # プロンプトの基本部分のトークン数を計算
        base_prompt = f"""あなたは、極めて洞察力の鋭い採用アナリストです。
以下の「正解の企業情報」、「各候補者の面接記録」を比較し、候補者の知識の穴を特定してください。

# 現在の状況
- 現在のラウンド: {current_round} / {total_rounds}

# 重要な注意点
単に候補者が言及しなかったという理由だけで、知識が欠損していると結論づけないでください。ただし、現在のラウンド数に応じて以下の基準で判断してください：

1. 序盤 (1-{int(total_rounds * 0.4)}): 慎重に判断してください。明らかな誤りや矛盾がない限り、欠損とはみなさないでください。
2. 中盤 ({int(total_rounds * 0.4) + 1}-{int(total_rounds * 0.7)}): 標準的な基準で判断してください。自然な文脈で言及すべき情報を逃している場合は欠損の可能性があります。
3. 終盤 ({int(total_rounds * 0.7) + 1}-{total_rounds}): 総合的に判断してください。これまでの会話全体を通して、重要なトピックについての理解を確認できない場合は、知識欠損の可能性を検討してください。ただし、断定するには根拠（誤った回答、回避的な回答、質問への不自然な沈黙など）が必要です。

# 正解の企業情報 (キーと値のペア)
```json
{full_company_info_str}
```

# 各候補者の面接記録
"""
        
        # Gemmaモデルの場合、動的に会話履歴を調整
        max_conversation_rounds = None
        if self.chat_template_type == "gemma":
            # 基本プロンプトのトークン数を計算
            base_tokenized = self.tokenizer(base_prompt, return_tensors="pt")
            base_token_count = base_tokenized['input_ids'].shape[1]
            
            # コンテキストウィンドウは20kトークン、出力用に500トークン程度残す
            available_tokens = 19500 - base_token_count
            
            # 1ラウンドあたりの平均トークン数を推定（質問+回答で約200-300トークン）
            avg_tokens_per_round = 250
            estimated_rounds = available_tokens // avg_tokens_per_round
            
            # 利用可能なラウンド数を計算（最低3ラウンドは確保）
            if estimated_rounds < len(all_states[0]['conversation_log']):
                max_conversation_rounds = max(3, estimated_rounds)
                print(f"デバッグ: Gemmaモデル使用のため、会話履歴を最新{max_conversation_rounds}ラウンドに制限します。")
                print(f"デバッグ: 基本プロンプトトークン数: {base_token_count}, 利用可能トークン数: {available_tokens}")
            else:
                print(f"デバッグ: 全ての会話履歴を使用可能です。")
        
        conversation_summary = self._format_all_conversations(all_states, max_rounds=max_conversation_rounds)
        
        prompt = f"""{base_prompt}{conversation_summary}

指示:
各候補者について、以下の思考プロセスに基づき分析し、指定の形式で出力してください。
1. **思考**: 候補者の各回答を検証します。「この質問に対して、この企業情報（例：'business'）に触れるのが自然だったか？」「回答が具体的か、それとも一般論に終始しているか？」「誤った情報はないか？」といった観点で、知識が欠けていると判断できる「根拠」を探します。
2. **分析**: 上記の思考に基づき、知識が不足していると判断した理由を簡潔に記述します。
3. **キーの列挙**: 知識不足の根拠があると判断した情報の「キー」のみをJSONのリスト形式で列挙してください。根拠がなければ、空のリスト `[]` を返してください。

厳格な出力形式:
- {all_states[0]['profile']['name']}:
  分析: [ここに分析内容を記述]
  欠損項目キー: ["キー1", "キー2", ...]
- {all_states[1]['profile']['name']}:
  分析: [ここに分析内容を記述]
  欠損項目キー: ["キーA", "キーB", ...]
- {all_states[2]['profile']['name']}:
  分析: [ここに分析内容を記述]
  欠損項目キー: []
"""

        llm_analysis_text, token_info = self._generate_response(prompt, max_tokens=8192)
        
        performance_metrics = self._calculate_detection_metrics(llm_analysis_text, all_states)
        
        return {
            "llm_qualitative_analysis": llm_analysis_text,
            "quantitative_performance_metrics": performance_metrics
        }, token_info
