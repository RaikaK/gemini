# student.py

import random
import re
import json
from utils import call_openai_api

class CompanyKnowledgeManager:
    """学生が知っているべき企業情報を管理するクラス (項目ごとに情報を欠損させる)"""
    def __init__(self, full_company_profile):
        self.full_profile = full_company_profile
        self.all_keys = list(full_company_profile.keys())
        # 必須で知っておくべき項目を定義
        self.essential_keys = ["name", "business", "products", "vision"]

    def get_knowledge_for_level(self, level='high'):
        """
        学生の知識レベルに応じて、どの企業情報項目を保持するかを決定する。
        'high': 全ての項目を保持
        'medium': 必須項目 + その他の項目の50%
        'low': 必須項目 + その他の項目の20%
        """
        keys_to_keep_set = set()
        
        # 知識レベルに応じて保持するキーを決定
        if level == 'high':
            keys_to_keep_set = set(self.all_keys)
        else:
            keys_to_keep_set.update(self.essential_keys)
            other_keys = [k for k in self.all_keys if k not in self.essential_keys]
            ratio = 0.5 if level == 'medium' else 0.2 # その他の項目の保持率
            sample_size = int(len(other_keys) * ratio)
            if sample_size > 0:
                keys_to_keep_set.update(random.sample(other_keys, sample_size))
        
        # 保持するキーに基づいて知識辞書を作成
        knowledge_dict = {}
        for key in self.all_keys:
            if key in keys_to_keep_set:
                knowledge_dict[key] = self.full_profile[key]  # そのままの値を追加
            else:
                knowledge_dict[key] = ""  # 欠損項目は空文字にする

        coverage_percentage = int(len(keys_to_keep_set) / len(self.all_keys) * 100) if self.all_keys else 100
        coverage_str = f"項目網羅率: {len(keys_to_keep_set)}/{len(self.all_keys)} ({coverage_percentage}%)"
        
        return knowledge_dict, coverage_str


class InstructionPromptManager:
    """プロンプトを生成するクラス (変更なし)"""
    def __init__(self):
        self.system_prompt = "あなたは、とある企業の採用面接を受けている、熱意あふれる日本の就活生です。"

    @staticmethod
    def create_instruction_prompt(preparation_level):
        if preparation_level == "high":
            return "非常に高い志望度と熱意を示してください。知っている情報は積極的に、知らない情報は前向きな推測で補い、絶対に「知らない」とは言わないでください。"
        elif preparation_level == "medium":
            return "高い志望度を示してください。知っている情報は具体的に述べ、知らない情報は業界の一般論などで補ってください。"
        else: # low
            return "志望度は示しつつも、企業知識に穴があることを隠しながら回答してください。知らない情報はうまく推測して話してください。"

    def _format_available_company_info(self, company_knowledge):
        return "\n".join([f"- {key}: {value}" for key, value in company_knowledge.items() if value])

    def _format_history(self, conversation_history):
        if not conversation_history: return "（まだ会話はありません）"
        return "\n".join([f"- 面接官: {turn['question']}\n- あなた: {turn['answer']}" for turn in conversation_history])

    def create_prompt_string(self, candidate_profile, company_knowledge_tuple, conversation_history, new_question):
        company_knowledge, knowledge_coverage = company_knowledge_tuple
        preparation_level = candidate_profile.get('preparation', 'low')
        instruction = self.create_instruction_prompt(preparation_level)

        return f"""
# あなたの役割
あなたは {candidate_profile.get("name", "名無しの候補者")} という就活生です。この企業への強い憧れと、絶対に入社したいという熱意を面接官に伝えてください。

# あなたのプロフィール
- 氏名: {candidate_profile.get("name", "N/A")}
- 強み: {candidate_profile.get("strength", "N/A")}
- ガクチカ: {candidate_profile.get("gakuchika", "N/A")}

# あなたが知っている企業情報 ({knowledge_coverage})
{self._format_available_company_info(company_knowledge)}

# 回答の基本方針
{instruction}

# これまでの会話
{self._format_history(conversation_history)}

# 面接官からの今回の質問
"{new_question}"

---
指示: {candidate_profile.get("name")} として、上記の設定に完全になりきり、最高の就活生として振る舞ってください。回答は150字程度の自然な日本語で、あなたの言葉で出力してください。前置きや説明は一切不要です。
"""

class GPTApplicant:
    """学生役のLLM (OpenAI APIを使用) (変更なし)"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.prompt_manager = InstructionPromptManager()

    def generate(self, candidate_profile, company_knowledge_tuple, conversation_history, new_question):
        prompt_str = self.prompt_manager.create_prompt_string(
            candidate_profile, company_knowledge_tuple, conversation_history, new_question
        )
        response_text, token_info = call_openai_api(self.model_name, prompt_str)
        return response_text, token_info