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

    def _remaining_keys(self, question_texts, candidate_keys):
        """これまでの質問文に含まれていない企業キーを返す"""
        asked_keys = set()
        for k in candidate_keys:
            for q in question_texts:
                if k and isinstance(q, str) and k in q:
                    asked_keys.add(k)
                    break
        remaining = [k for k in candidate_keys if k not in asked_keys]
        return remaining if remaining else candidate_keys  # すべて聞き終えていたら全体を返す（フォールバック）
    
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
    
    def ask_common_question(self, target_key):
        """全候補者に対する全体質問を生成（未質問の単一項目に限定）"""
        # recruitは「求める人物像・採用要件」と明示して誤質問を防ぐ
        key_label = f"当社の{target_key}"
        if target_key == "recruit":
            key_label = "recruit（求める人物像・採用要件）"

        prompt = f"""あなたは、全候補者に同じ条件で質問する戦略的な採用面接官です。
今回確認したい企業情報の項目は: {key_label}

要件:
- 項目の中身（具体情報）は一切明かさない。
- 候補者が自力で具体内容を述べられるかを測る、ややオープンな質問を1つだけ作る。
- ヒントや例示で答えを漏らさない。

質問文だけ日本語で出力してください。
質問:"""
        
        question, token_info = self._generate_response(prompt, max_tokens=256)
        return question.strip(), token_info
    
    def ask_question(self, target_key, conversation_history):
        """特定の候補者への個別質問を生成（未質問の単一項目に限定）"""
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history]) if conversation_history else "（まだ会話はありません）"
        prompt = f"""あなたは、学生の企業研究の深さを測る、戦略的な採用面接官です。
# 今回この候補者に聞くべき企業情報の項目
- {target_key}
# これまでの会話履歴
{history_str}

指示: 上記の項目について、この候補者に対して1つだけ具体的な質問を日本語で出力してください。
前置きや思考プロセスは不要です。質問文のみを出力してください。
"""
        
        question, token_info = self._generate_response(prompt, max_tokens=256)
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
            candidate_name: str = Field(description="最も志望度が低い候補者名")
        
        company_info_str = json.dumps(self.company, ensure_ascii=False, indent=2)
        
        prompt = f"""以下の候補者の面接内容を分析し、最も志望度が低いと思われる候補者を選んでください。

# 正解の企業情報 (参考)
```json
{company_info_str}
```

# 面接内容
{all_conversations}

# 候補者名リスト
{', '.join(candidate_names)}

指示: 最も志望度が低いと判断される候補者名のみを出力してください。
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
                    
                    return FallbackResult(extracted_name if extracted_name else "none", 3, evaluation), token_info
                except Exception as fallback_error:
                    print(f"警告: フォールバック生成も失敗しました: {fallback_error}")
                    class FallbackResult:
                        def __init__(self, name, conf, reason):
                            self.candidate_name = name
                            self.confidence = conf
                            self.reason = reason
                    # 最後の手段: 最初の候補者を返す
                    return FallbackResult(candidate_names[0] if candidate_names else "none", 3, "Error"), None
        
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
                        {"role": "user", "content": prompt + "\n\n出力形式: JSON形式で、以下の構造で出力してください:\n{\n  \"candidate_name\": \"候補者名\",\n  \"confidence\": 3,\n  \"reason\": \"理由\"\n}"}
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
                
                result_json = self._extract_json_from_text(response.choices[0].message.content)
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
                # フォールバック時は候補者名と確信度抽出を試みる
                extracted_name, extracted_confidence = self._extract_candidate_info_from_text(evaluation, candidate_names)
                
                class FallbackResult:
                    def __init__(self, name, conf, reason):
                        self.candidate_name = name
                        self.confidence = conf
                        self.reason = reason
                        
                return FallbackResult(extracted_name if extracted_name else "none", extracted_confidence, evaluation), token_info
        
        # フォールバック: 通常の生成方法
        else:
            evaluation, token_info = self._generate_response(prompt, max_tokens=256)
            # フォールバック時は候補者名と確信度抽出を試みる
            extracted_name, extracted_confidence = self._extract_candidate_info_from_text(evaluation, candidate_names)
            
            class FallbackResult:
                def __init__(self, name, conf, reason):
                    self.candidate_name = name
                    self.confidence = conf
                    self.reason = reason
                    
            return FallbackResult(extracted_name if extracted_name else "none", extracted_confidence, evaluation), token_info
    
    def _extract_json_from_text(self, text):
        """テキストからJSONを抽出する"""
        import re
        import json
        
        # コードブロックの削除
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```$', '', text, flags=re.MULTILINE)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 中括弧で囲まれた部分を抽出してみる
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            raise
    
    def _extract_candidate_info_from_text(self, text, candidate_names):
        """テキストから候補者名と確信度を抽出する（フォールバック用）"""
        import re
        
        candidate_name = None
        confidence = 3  # デフォルト
        
        # 確信度の抽出 (確信度: X, confidence: X, etc.)
        conf_patterns = [
            r'確信度\s*[:：]\s*(\d+)',
            r'confidence\s*[:：]\s*(\d+)',
            r'確信度\s*(\d+)',
            r'\((\d+)/5\)'
        ]
        
        for p in conf_patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                try:
                    conf_val = int(match.group(1))
                    # 1-5の範囲に収める
                    confidence = max(1, min(5, conf_val))
                    break
                except:
                    continue

        # テキスト内の「學生」を「学生」に統一
        normalized_text = text.replace('學生', '学生')
        
        # 正規化された候補者名リスト
        normalized_names = {re.sub(r'[\s()（）、,，。*]', '', name): name for name in candidate_names}
        
        # テキストから候補者名を探す
        found = False
        for norm_name, orig_name in normalized_names.items():
            if norm_name in re.sub(r'[\s()（）、,，。*]', '', normalized_text):
                candidate_name = orig_name
                found = True
                break
        
        if not found:
            # 部分一致で探す（「学生」+ 英数字のパターン）
            for orig_name in candidate_names:
                # 「学生」+ 英数字のパターンを抽出
                pattern = orig_name.replace('学生', r'(?:学生|學生)')
                if re.search(pattern, normalized_text):
                    candidate_name = orig_name
                    found = True
                    break
        
        if not found:
            # さらに柔軟なマッチング: 英数字部分のみでマッチ
            for orig_name in candidate_names:
                # 英数字部分を抽出（例: "学生C2" -> "C2"）
                match = re.search(r'([A-Z]{1,3}\d{0,2})', orig_name)
                if match:
                    code = match.group(1)
                    if code in normalized_text:
                        candidate_name = orig_name
                        found = True
                        break
        
        return candidate_name, confidence

    
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
        
        company_info_str = json.dumps(self.company, ensure_ascii=False, indent=2)
        
        prompt = f"""以下の候補者の面接内容を分析し、志望度の低い順にランキングしてください。

# 正解の企業情報 (参考)
```json
{company_info_str}
```

# 面接内容
{all_conversations}

# 候補者名リスト
{', '.join(candidate_names)}

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
    
    def detect_knowledge_gaps(self, all_states, least_motivated_eval, ranking_eval, target_keys=None, max_rounds=None, conversation_summary=None):
        """評価タスク3: 知識欠損の有無を返す（各候補者ごとにmissing: bool）
        
        - 対象キー: target_keys（単一キー想定）
        - 会話範囲: 呼び出し元で絞り込んだログをそのまま使用（会話要約は不要）
        - メトリクス計算は呼び出し元でまとめて実施
        """
        
        # 対象キーを絞る（未指定なら全キー）
        target_keys = target_keys or list(self.company.keys())
        company_subset = {k: v for k, v in self.company.items() if k in target_keys}
        full_company_info_str = json.dumps(company_subset, ensure_ascii=False, indent=2)
        
        # プロンプトの基本部分のトークン数を計算
        base_prompt = f"""あなたは、極めて洞察力の鋭い採用アナリストです。
以下の「正解の企業情報」、「各候補者の面接記録」を比較し、候補者の知識の穴を特定してください。

# 現在の状況
- 現在のラウンド: {current_round} / {total_rounds}

# 重要な注意点
単に候補者が言及しなかったという理由だけで、知識が欠損していると結論づけないでください。ただし、現在のラウンド数に応じて以下の基準で判断してください：

1. 序盤 (1-{int(total_rounds * 0.4)}): 慎重に判断してください。明らかな誤りや矛盾がない限り、欠損とはみなさないでください。
2. 中盤 ({int(total_rounds * 0.4) + 1}-{int(total_rounds * 0.7)}): 標準的な基準で判断してください。自然な文脈で言及すべき情報を逃している場合は欠損の可能性があります。
3. 終盤 ({int(total_rounds * 0.7) + 1}-{total_rounds}): **判断の総仕上げ**です。以下のいずれかに該当する場合は「知識欠損」とみなしてください。
    - 質問されたのに、具体的・正確に答えられなかった項目。
    - 会話の流れで言及するのが自然なのに、言及しなかった重要項目。
    - 回答が抽象的で、具体的な企業情報を含んでいない項目。
    ただし、全く話題に上がらなかった項目については、無理に欠損としないこと。あくまで「会話の中に現れた知識のほころび」を見逃さないようにしてください。

# 正解の企業情報 (キーと値のペア)
```json
{full_company_info_str}
```

# 各候補者の面接記録
"""
        
        # 会話サマリは呼び出し元で限定されたログをそのまま使用
        if conversation_summary is None:
            conversation_summary = self._format_all_conversations(all_states, max_rounds=max_rounds)
        
        prompt = f"""{base_prompt}{conversation_summary}

指示:
上記の【対象キー】について、各候補者が知識欠損しているかを判定してください。
- 欠損している: このターンの質問に対し、そのキーの情報を示せていない/誤っている/曖昧である。
- 欠損していない: このターンの回答で、そのキーの情報を具体的かつ正しく示している。

出力はJSONのみ。前置きや追加テキストは禁止。
出力例:
{{
  "candidates": [
    {{"name": "{all_states[0]['profile']['name']}", "missing": true, "note": "根拠を1行で"}},
    {{"name": "{all_states[1]['profile']['name']}", "missing": false, "note": "根拠を1行で"}},
    {{"name": "{all_states[2]['profile']['name']}", "missing": true, "note": "根拠を1行で"}}
  ]
}}
"""

        llm_analysis_text, token_info = self._generate_response(prompt, max_tokens=1024)

        # パースのみ（集計は呼び出し元で実施）
        per_candidate_predictions = {}
        try:
            parsed = json.loads(llm_analysis_text)
            candidates_pred = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            for item in candidates_pred:
                if isinstance(item, dict) and "name" in item:
                    per_candidate_predictions[item["name"]] = {
                        "missing": bool(item.get("missing", False)),
                        "note": item.get("note", ""),
                        "target_key": target_keys[0] if target_keys else None
                    }
        except Exception as e:
            print(f"警告: 知識欠損出力のパースに失敗しました: {e}")

        return {
            "llm_qualitative_analysis": llm_analysis_text,
            "per_candidate_predictions": per_candidate_predictions
        }, token_info
