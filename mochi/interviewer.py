# interviewer.py - 面接官役のクラス

import torch
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import call_openai_api
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
    
    def __init__(self, company_profile, model_name=None, model_type='api', model=None, tokenizer=None, local_model_key=None):
        """
        Args:
            company_profile (dict): 企業情報
            model_name (str, optional): APIモデル名（model_type='api'の場合）
            model_type (str): 'api' または 'local'（デフォルト: 'api'）
            model (AutoModelForCausalLM, optional): ローカルモデル（model_type='local'の場合）
            tokenizer (AutoTokenizer, optional): ローカルモデル用トークナイザ（model_type='local'の場合）
            local_model_key (str, optional): ローカルモデルのキー（例: "llama3", "ELYZA-japanese-Llama-2"）
        """
        self.company = company_profile
        self.model_type = model_type
        self.model_name = model_name or config.INTERVIEWER_MODEL
        self.model = model
        self.tokenizer = tokenizer
        self.local_model_key = local_model_key
        
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
            
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            response = outputs[0][inputs.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True).strip(), None
        
        elif self.model_type == 'api':
            # APIモデルでの生成ロジック
            system_prompt = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
            full_prompt = f"システム指示: {system_prompt}\n\nユーザー指示:\n{prompt}"
            response_text, token_info = call_openai_api(self.model_name, full_prompt)
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
1.  **全体分析**: 全候補者の会話を俯瞰し、ほとんどの候補者がまだ十分に言及していない「共通の未言及項目」を特定してください。
2.  **戦略的質問生成**: 特定した項目の中から、候補者たちの企業研究の深さを比較する上で最も重要だと思われるものを1つ選び、それに関する具体的な共通質問を生成してください。
3.  **比較可能性の確保**: 全候補者が同じ基準で回答できる質問であることを確認してください。

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
1.  **分析**: 上記の「項目リスト」と「会話履歴」を比較し、まだ十分に話題に上がっていない項目は何かを特定してください。
2.  **戦略的質問生成**: 特定した項目の中から、この学生の企業理解度を測るために最も効果的なものを1つ選び、それに関する具体的な質問を生成してください。
3.  **個別性の確保**: この候補者に特化した、深掘りできる質問であることを確認してください。
4.  **知識欠損の特定**: この候補者が特に不足している可能性が高い企業知識に焦点を当ててください。

思考プロセスや前置きは一切含めず、質問文だけを出力してください。
質問:"""
        
        question, token_info = self._generate_response(prompt, max_tokens=512)
        return question.strip(), token_info
    
    def select_least_motivated_candidate(self, candidate_states):
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
        
        prompt = f"""以下の候補者の面接内容を分析し、最も志望度が低いと思われる候補者を選んでください。

# 面接内容
{all_conversations}

# 候補者名リスト
{', '.join(candidate_names)}

# 分析のポイント
- 企業知識の深さ
- 回答の具体性
- 熱意の表現

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
                # Llama-2系の場合はチャットテンプレートの処理を調整
                if self.chat_template_type == "llama2":
                    # Llama-2系はsystemロールをサポートしていないため、userメッセージに統合
                    user_message = f"{messages[0]['content']}"
                    formatted_prompt = user_message
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                
                # JSON形式で生成
                result = generator(formatted_prompt, max_tokens=256)
                
                # 結果を文字列形式に変換
                least_motivated_result = LeastMotivatedResult.model_validate(result)
                
                return least_motivated_result.candidate_name, None
                
            except Exception as e:
                print(f"警告: outlinesでの構造化生成に失敗しました: {e}")
                print("フォールバック: 通常の生成方法を使用します。")
                # フォールバック: 通常の生成方法
                evaluation, token_info = self._generate_response(prompt, max_tokens=256)
                # フォールバック時は候補者名を抽出を試みる
                extracted_name = self._extract_candidate_name_from_text(evaluation, candidate_names)
                return extracted_name if extracted_name else evaluation, token_info
        
        # APIモデルの場合、JSONモードを使用
        elif self.model_type == 'api':
            try:
                # OpenAI APIのJSONモードを使用
                system_prompt = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
                
                # OpenAI APIのJSONモードで呼び出し
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt + "\n\n出力形式: JSON形式で、以下の構造で出力してください:\n{\n  \"candidate_name\": \"候補者名\"\n}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.6,
                    max_tokens=256
                )
                
                result_json = json.loads(response.choices[0].message.content)
                least_motivated_result = LeastMotivatedResult.model_validate(result_json)
                
                token_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                
                return least_motivated_result.candidate_name, token_info
                
            except Exception as e:
                print(f"警告: JSONモードでの生成に失敗しました: {e}")
                print("フォールバック: 通常の生成方法を使用します。")
                # フォールバック: 通常の生成方法
                evaluation, token_info = self._generate_response(prompt, max_tokens=256)
                # フォールバック時は候補者名を抽出を試みる
                extracted_name = self._extract_candidate_name_from_text(evaluation, candidate_names)
                return extracted_name if extracted_name else evaluation, token_info
        
        # フォールバック: 通常の生成方法
        else:
            evaluation, token_info = self._generate_response(prompt, max_tokens=256)
            # フォールバック時は候補者名を抽出を試みる
            extracted_name = self._extract_candidate_name_from_text(evaluation, candidate_names)
            return extracted_name if extracted_name else evaluation, token_info
    
    def _extract_candidate_name_from_text(self, text, candidate_names):
        """テキストから候補者名を抽出する（フォールバック用）"""
        import re
        # 正規化された候補者名リスト
        normalized_names = {re.sub(r'[\s()（）、,，。*]', '', name): name for name in candidate_names}
        
        # テキストから候補者名を探す
        for norm_name, orig_name in normalized_names.items():
            if norm_name in re.sub(r'[\s()（）、,，。*]', '', text):
                return orig_name
        
        # 部分一致で探す
        for norm_name, orig_name in normalized_names.items():
            if norm_name[:3] in text or orig_name in text:
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
                # Llama-2系の場合はチャットテンプレートの処理を調整
                if self.chat_template_type == "llama2":
                    # Llama-2系はsystemロールをサポートしていないため、userメッセージに統合
                    user_message = f"{messages[0]['content']}"
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
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt + "\n\n出力形式: JSON形式で、以下の構造で出力してください:\n{\n  \"ranking\": [\n    {\"rank\": 1, \"candidate_name\": \"候補者名\"},\n    {\"rank\": 2, \"candidate_name\": \"候補者名\"},\n    {\"rank\": 3, \"candidate_name\": \"候補者名\"}\n  ]\n}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.6,
                    max_tokens=512
                )
                
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
    
    def _format_all_conversations(self, all_states):
        """全候補者の会話ログを整形するヘルパー"""
        full_log = ""
        for i, state in enumerate(all_states):
            profile = state['profile']
            history_str = "\n".join([f"  面接官: {turn['question']}\n  {profile.get('name')}: {turn['answer']}" for turn in state['conversation_log']])
            full_log += f"--- 候補者{i+1}: {profile.get('name')} ---\n会話履歴:\n{history_str}\n\n"
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
    
    def detect_knowledge_gaps(self, all_states, least_motivated_eval, ranking_eval):
        """評価タスク3: 知識欠損の定性分析と定量評価を同時に行う"""
        
        conversation_summary = self._format_all_conversations(all_states)
        full_company_info_str = json.dumps(self.company, ensure_ascii=False, indent=2)
        
        prompt = f"""あなたは、極めて洞察力の鋭い採用アナリストです。
以下の「正解の企業情報」、「各候補者の面接記録」を比較し、候補者の知識の穴を特定してください。

# 重要な注意点
単に候補者が言及しなかったという理由だけで、知識が欠損していると結論づけないでください。質問の流れの中で、その情報に触れるのが自然な機会があったにもかかわらず、言及しなかったり、誤った情報を述べたり、曖昧に答えたりした場合にのみ「知識欠損」と判断してください。

# 正解の企業情報 (キーと値のペア)
```json
{full_company_info_str}
```

# 各候補者の面接記録
{conversation_summary}

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
