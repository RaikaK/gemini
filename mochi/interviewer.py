# interviewer.py - 面接官役のクラス

import torch
from utils import call_openai_api
import config

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
        """最も志望度が低いと思われる候補者を選定"""
        
        # 各候補者の会話ログを整形
        candidates_info = []
        for i, state in enumerate(candidate_states):
            name = state['profile']['name']
            conversations = []
            for turn in state['conversation_log']:
                conversations.append(f"Q: {turn['question']}\nA: {turn['answer']}")
            conv_str = "\n".join(conversations)
            candidates_info.append(f"候補者{i+1}: {name}\n{conv_str}")
        
        all_conversations = "\n\n".join(candidates_info)
        
        prompt = f"""あなたは採用面接官です。以下の候補者の面接内容を分析し、最も志望度が低いと思われる候補者を選んでください。

# 面接内容
{all_conversations}

# 分析のポイント
- 企業知識の深さ
- 回答の具体性
- 熱意の表現

指示: 最も志望度が低いと判断される候補者の名前と、その理由を簡潔に述べてください。
"""
        
        evaluation, token_info = self._generate_response(prompt)
        return evaluation, token_info
    
    def rank_candidates_by_motivation(self, candidate_states):
        """候補者を志望度順にランキング"""
        
        # 各候補者の会話ログを整形
        candidates_info = []
        for i, state in enumerate(candidate_states):
            name = state['profile']['name']
            conversations = []
            for turn in state['conversation_log']:
                conversations.append(f"Q: {turn['question']}\nA: {turn['answer']}")
            conv_str = "\n".join(conversations)
            candidates_info.append(f"候補者{i+1}: {name}\n{conv_str}")
        
        all_conversations = "\n\n".join(candidates_info)
        
        prompt = f"""あなたは採用面接官です。以下の候補者の面接内容を分析し、志望度の低い順にランキングしてください。

# 面接内容
{all_conversations}

# ランキングの基準
- 企業知識の深さ（調べている度合い）
- 回答の具体性
- 熱意の表現

指示: 志望度が低い順（1位が最も志望度が低い）にランキングし、以下の形式で出力してください：
1位: [候補者名]
2位: [候補者名]
3位: [候補者名]
"""
        
        ranking, token_info = self._generate_response(prompt)
        return ranking, token_info
