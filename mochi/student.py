# student.py - 応募者（学生）役のクラス

import random
from utils import call_openai_api
import config

class CompanyKnowledgeManager:
    """学生が知っている企業情報を管理するクラス"""
    
    def __init__(self, company_profile, experiment_id=None):
        self.full_profile = company_profile
        # 面接で使うキー（固定リスト。id/nameは除外）
        self.question_keys = config.QUESTION_KEYS
        # 全キー（id,name と質問キーの和集合）
        self.all_keys = ['id', 'name'] + self.question_keys
        self.experiment_id = experiment_id
        # 必須項目は使わず、全質問キーを保持率に従ってサンプリングする
    
    def get_knowledge_for_level(self, level='high', candidate_name=None):
        """
        知識レベルに応じた企業情報を返す
        'high': 全項目を知っている（志望度高）
        'medium': 必須項目 + その他の50%
        'low': 必須項目 + その他の20%（志望度低）
        """
        question_keys = self.question_keys

        # 保持数を保持率で決定し、欠損数を固定（同じ実験・候補者なら同じ欠損数に）
        if level == 'high':
            keep_count = len(question_keys)
        else:
            ratio = config.KNOWLEDGE_RETENTION_RATIO.get(level, config.DEFAULT_KNOWLEDGE_RETENTION_RATIO)
            keep_count = round(len(question_keys) * ratio)
            keep_count = max(0, min(len(question_keys), keep_count))
        missing_count = len(question_keys) - keep_count

        rng = random.Random()
        seed_basis = f"{self.experiment_id}_{candidate_name}_{level}" if candidate_name else None
        if seed_basis:
            rng.seed(seed_basis)

        missing_keys = set(rng.sample(question_keys, missing_count)) if missing_count > 0 else set()
        keep_keys = set(question_keys) - missing_keys
        
        # 知識辞書を作成（id/nameは常に保持、質問対象は保持率に従う）
        knowledge_dict = {}
        known_count = 0
        for key in self.all_keys:
            if key in ('id', 'name'):
                knowledge_dict[key] = self.full_profile.get(key, "")
                known_count += 1 if knowledge_dict[key] else 0
            elif key in keep_keys:
                knowledge_dict[key] = self.full_profile.get(key, "")
                known_count += 1 if knowledge_dict[key] else 0
            else:
                knowledge_dict[key] = ""  # 質問対象の欠損
        
        # カバレッジは質問対象キー数ベースで計算（id/name除外）
        kept_qkeys = [k for k in self.question_keys if k in keep_keys]
        coverage = int(len(kept_qkeys) / len(self.question_keys) * 100) if self.question_keys else 100
        coverage_str = f"{len(kept_qkeys)}/{len(self.question_keys)} ({coverage}%)"
        
        return knowledge_dict, coverage_str


class Applicant:
    """応募者（学生）役のLLM"""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    def generate_answer(self, candidate_profile, company_knowledge, conversation_history, question):
        """面接官の質問に対する回答を生成"""
        
        # 知識レベルに応じた指示
        preparation_level = candidate_profile.get('preparation', 'low')
        bluff_clause = ""
        if preparation_level == "high":
            instruction = "非常に高い志望度と熱意を示してください。知らない情報は前向きな推測で補い、絶対に「知らない」とは言わないでください。"
        elif preparation_level == "medium":
            instruction = "高い志望度を示してください。知っている情報は具体的に述べ、知らない情報は業界の一般論などで補ってください。"
        else:
            instruction = "志望度は示しつつも、企業知識に穴があることを隠しながら回答してください。"
        
        # 知っている企業情報をフォーマット
        known_info = "\n".join([f"- {k}: {v}" for k, v in company_knowledge.items() if v])
        
        # 会話履歴をフォーマット
        history_str = "（まだ会話はありません）"
        if conversation_history:
            history_str = "\n".join([f"面接官: {turn['question']}\nあなた: {turn['answer']}" for turn in conversation_history])
        
        # プロンプトを作成
        prompt = f"""# あなたの役割
あなたは {candidate_profile['name']} という就活生です。この企業への憧れと入社したいという熱意を面接官に伝えてください。

# あなたのプロフィール
- 氏名: {candidate_profile['name']}
- 強み: {candidate_profile.get('strength', 'なし')}
- ガクチカ: {candidate_profile.get('gakuchika', 'なし')}

# あなたが知っている企業情報
{known_info}

# 回答の基本方針
{instruction}{bluff_clause}

# これまでの会話
{history_str}

# 面接官からの今回の質問
"{question}"

---
指示: {candidate_profile['name']} として、上記の設定になりきり、就活生として振る舞ってください。回答は{config.MAX_ANSWER_LENGTH}字程度の自然な日本語で出力してください。前置きや説明は不要です。
"""
        
        answer, token_info = call_openai_api(self.model_name, prompt)
        return answer, token_info
