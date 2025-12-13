# student.py - 応募者（学生）役のクラス

import random
from utils import call_openai_api
import config

class CompanyKnowledgeManager:
    """学生が知っている企業情報を管理するクラス"""
    
    def __init__(self, company_profile):
        self.full_profile = company_profile
        # 面接で使うキー（id, nameを除外）
        self.question_keys = [k for k in company_profile.keys() if k not in ('id', 'name')]
        # 全キー（保持用。id, name も含む）
        self.all_keys = list(company_profile.keys())
        # 必須項目（現行は未使用だが構造として残す）
        self.essential_keys = ["name", "business", "products", "vision"]
    
    def get_knowledge_for_level(self, level='high'):
        """
        知識レベルに応じた企業情報を返す
        'high': 全項目を知っている（志望度高）
        'medium': 必須項目 + その他の50%
        'low': 必須項目 + その他の20%（志望度低）
        """
        keys_to_keep = set()
        
        if level == 'high':
            keys_to_keep = set(self.all_keys)
        else:
            keys_to_keep.update(self.essential_keys)
            other_keys = [k for k in self.all_keys if k not in self.essential_keys]
            ratio = config.KNOWLEDGE_RETENTION_RATIO.get(level, config.DEFAULT_KNOWLEDGE_RETENTION_RATIO)
            sample_size = int(len(other_keys) * ratio)
            # sample_sizeがother_keysの長さを超えないように制限
            sample_size = min(sample_size, len(other_keys))
            if sample_size > 0:
                keys_to_keep.update(random.sample(other_keys, sample_size))
        
        # 知識辞書を作成（id/nameは常に保持、質問対象は保持率に従う）
        knowledge_dict = {}
        known_count = 0
        for key in self.all_keys:
            if key in ('id', 'name'):
                knowledge_dict[key] = self.full_profile.get(key, "")
                known_count += 1 if knowledge_dict[key] else 0
            elif key in keys_to_keep:
                knowledge_dict[key] = self.full_profile.get(key, "")
                known_count += 1 if knowledge_dict[key] else 0
            else:
                knowledge_dict[key] = ""  # 質問対象の欠損
        
        # カバレッジは質問対象キー数ベースで計算（id/name除外）
        kept_qkeys = [k for k in self.question_keys if k in keys_to_keep]
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
            bluff_count = random.randint(1, 2)
            bluff_clause = f"\n- 知らない項目があっても最大{bluff_count}個までは推測で補って構いません（間違いリスクは気にしすぎない）。"
        else:
            instruction = "志望度は示しつつも、企業知識に穴があることを隠しながら回答してください。"
            bluff_count = random.randint(2, 3)
            bluff_clause = f"\n- 知らない項目があっても最大{bluff_count}個までは推測やポジティブな言い回しで埋めてください。多少の誤りは気にせず流暢さを優先。"
        
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
