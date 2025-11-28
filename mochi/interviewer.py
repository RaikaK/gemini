# interviewer.py - 面接官役のクラス

import torch
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import call_openai_api
import config

# outlinesのインポート（オプション）
try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    print("警告: outlinesモジュールが見つかりません。構造化出力は使用できません。")

class Interviewer:
    """面接官役のLLM（ローカルモデルとAPIモデルの両方に対応）"""
    
    def __init__(self, company_profile, model_name=None, model_type='api', model=None, tokenizer=None):
        """
        Args:
            company_profile (dict): 企業情報
            model_name (str, optional): APIモデル名（model_type='api'の場合）
            model_type (str): 'api' または 'local'（デフォルト: 'api'）
            model (AutoModelForCausalLM, optional): ローカルモデル（model_type='local'の場合）
            tokenizer (AutoTokenizer, optional): ローカルモデル用トークナイザ（model_type='local'の場合）
        """
        self.company = company_profile
        self.model_type = model_type
        self.model_name = model_name or config.INTERVIEWER_MODEL
        self.model = model
        self.tokenizer = tokenizer
        
        if self.model_type == 'local' and (not self.model or not self.tokenizer):
            raise ValueError("ローカルモデルタイプには 'model' と 'tokenizer' が必要です。")
    
    def _generate_response(self, prompt, max_tokens=512):
        """モデルタイプに応じて応答を生成する"""
        if self.model_type == 'local':
            # ローカルモデルでの生成ロジック
            messages = [
                {"role": "system", "content": "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"},
                {"role": "user", "content": prompt}
            ]
            
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            
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
    
    def ask_question(self, conversation_history, asked_questions):
        """候補者への質問を生成"""
        
        # 企業情報の項目リスト
        company_keys = list(self.company.keys())
        
        # これまでの質問履歴
        history_str = "（なし）"
        if asked_questions:
            history_str = "\n".join([f"- {q}" for q in asked_questions])
        
        prompt = f"""あなたは採用面接官です。候補者の企業理解度と志望度を測る質問をしてください。

# 企業情報の項目リスト
{company_keys}

# これまでに行った質問（重複しないように）
{history_str}

# 質問の目的
- 候補者が企業についてどれだけ調べているか確認する
- 志望度の高さを測る
- 企業理解の深さを探る

指示: 上記の目的に沿った効果的な質問を1つ生成してください。質問文のみを出力してください。前置きや説明は不要です。
"""
        
        question, token_info = self._generate_response(prompt)
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
                outlines_model = outlines.models.transformers(self.model, self.tokenizer)
                generator = outlines.generate.json(outlines_model, LeastMotivatedResult)
                
                messages = [
                    {"role": "system", "content": "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"},
                    {"role": "user", "content": prompt}
                ]
                
                # チャットテンプレートを適用してプロンプトを作成
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
                outlines_model = outlines.models.transformers(self.model, self.tokenizer)
                generator = outlines.generate.json(outlines_model, RankingResult)
                
                messages = [
                    {"role": "system", "content": "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"},
                    {"role": "user", "content": prompt}
                ]
                
                # チャットテンプレートを適用してプロンプトを作成
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
