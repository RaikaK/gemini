# interviewer.py - 面接官役のクラス

from utils import call_openai_api

class Interviewer:
    """面接官役のLLM"""
    
    def __init__(self, company_profile, model_name):
        self.company = company_profile
        self.model_name = model_name
    
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
        
        question, token_info = call_openai_api(self.model_name, prompt)
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
        
        evaluation, token_info = call_openai_api(self.model_name, prompt)
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
        
        ranking, token_info = call_openai_api(self.model_name, prompt)
        return ranking, token_info
