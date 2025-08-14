# 学生に関するファイル．
# 企業情報の穴抜きや応答生成を行うクラスを定義します。

import random
import re
import json

class CompanyKnowledgeManager:
    """学生が知っているべき企業情報を管理するクラス"""
    def __init__(self, full_company_profile):
        self.full_profile = full_company_profile
        self.all_keys = list(full_company_profile.keys())
        self.essential_keys = ["name", "business", "products", "vision"]

    def _punch_holes_in_string(self, text: str, percentage: float, placeholder: str = '_') -> str:
        """
        文字列を単語単位で穴抜きする。
        指定された割合の単語を '_' の連続に置換する。
        「」内の単語に対しては、穴抜き率を低減する。
        非単語文字（スペース、句読点など）は穴抜きしない。
        """
        if not isinstance(text, str) or not text:
            return text

        result_tokens = []
        
        segments = re.split(r'(「.*?」)', text)
        
        for segment in segments:
            is_quoted = segment.startswith('「') and segment.endswith('」') and len(segment) > 1
            
            effective_percentage = (percentage / 3.0) if is_quoted else percentage
            
            inner_text = segment[1:-1] if is_quoted else segment
            
            tokens = re.findall(r'[一-龯ぁ-んァ-ヶ々a-zA-Z0-9]+|\s+|.', inner_text)
            
            punched_tokens = []
            for token in tokens:
                if re.fullmatch(r'[一-龯ぁ-んァ-ヶ々a-zA-Z0-9]+', token) and random.random() < effective_percentage:
                    punched_tokens.append(placeholder * len(token))
                else:
                    punched_tokens.append(token)
            
            processed_segment = "".join(punched_tokens)
            if is_quoted:
                result_tokens.append('「' + processed_segment + '」')
            else:
                result_tokens.append(processed_segment)
        
        return "".join(result_tokens)

    def get_knowledge_for_level(self, level='high'):
        """準備レベルに応じて、フィルタリングされ、穴の開いた企業情報と知識のカバレッジ率を返す"""
        keys_to_keep_set = set()
        if level == 'high':
            keys_to_keep_set = set(self.all_keys)
        else:
            keys_to_keep_set.update(self.essential_keys)
            other_keys = [k for k in self.all_keys if k not in self.essential_keys]
            
            ratio = 0.0
            if level == 'medium':
                ratio = 1.0
            elif level == 'low':
                ratio = 1.0
            
            sample_size = int(len(other_keys) * ratio)
            if sample_size > 0:
                keys_to_keep_set.update(random.sample(other_keys, sample_size))
        
        hole_percentage = 0.0
        if level == 'medium':
            hole_percentage = 0.2
        elif level == 'low':
            hole_percentage = 0.4

        knowledge_dict = {}
        for key in self.all_keys:
            if key in keys_to_keep_set:
                value = self.full_profile[key]
                if level != 'high' and key not in self.essential_keys:
                    knowledge_dict[key] = self._punch_holes_in_string(value, hole_percentage)
                else:
                    knowledge_dict[key] = value
            else:
                knowledge_dict[key] = "" 

        coverage_percentage = int(len(keys_to_keep_set) / len(self.all_keys) * 100) if self.all_keys else 100
        coverage_str = f"{len(keys_to_keep_set)}/{len(self.all_keys)}項目 ({coverage_percentage}%)"
        
        return knowledge_dict, coverage_str


class InstructionPromptManager:
    """Llama 3.1用のプロンプトを生成するクラス"""
    def __init__(self):
        self.system_prompt = "あなたは優秀な日本語AIアシスタントです。指示に従って適切に回答してください。"

    @staticmethod
    def create_instruction_prompt(preparation_level):
        """回答方針作成メソッド"""
        if preparation_level == "high":
            return """
- 非常に高い志望度と熱意を必ず示してください。
- 他の就活生に負けない強い意欲を表現してください。
- 知っている具体的な情報は積極的に言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを強く表現してください。"""
        elif preparation_level == "medium":
            return """
- 高い志望度と熱意を必ず示してください。
- 他の就活生に負けない意欲を表現してください。
- 知っている具体的な情報は積極的に言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを表現してください。
- 企業情報の不足している部分(〇〇)は推測しながら話してください。
"""
        else: # low
            return """
- そこそこ高い志望度と熱意を必ず示してください。
- 他の就活生に負けない意欲をなるべく表現してください。
- 知っている具体的な情報は言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを表現してください。
- 企業情報の不足している部分(〇〇)は推測しながら話してください。
"""

    def _format_available_company_info(self, company_knowledge):
        return "\n".join([f"- {key}: {value}" for key, value in company_knowledge.items() if value])

    def _format_history(self, conversation_history):
        if not conversation_history:
            return "（まだ会話はありません）"
        return "\n".join([f"- 面接官: {turn['question']}\n- あなた: {turn['answer']}" for turn in conversation_history])

    def create_messages(self, candidate_profile, company_knowledge_tuple, conversation_history, new_question):
        company_knowledge, knowledge_coverage = company_knowledge_tuple
        preparation_level = candidate_profile.get('preparation', 'low')
        instruction_prompt = self.create_instruction_prompt(preparation_level)

        user_content = f"""
あなたは {candidate_profile.get("name", "名無しの候補者")} という日本の就活生です。この企業に絶対に入社したく、面接官に志望度の高さを強くアピールしたいと考えています。
企業研究を熱心に行いましたが、情報収集には限界があります。知っている情報は具体的に、知らない情報は前向きな推測や一般論で補って回答してください。

# あなたのプロフィール
- 氏名: {candidate_profile.get("name", "N/A")}
- 大学: {candidate_profile.get("university", "N/A")}
- 強み: {candidate_profile.get("strength", "N/A")}
- ガクチカ: {candidate_profile.get("gakuchika", "N/A")}
- MBTI: {candidate_profile.get("mbti", "N/A")}

# あなたが調べて得た企業情報（{knowledge_coverage}）
{self._format_available_company_info(company_knowledge)}

# 回答の重要な方針
{instruction_prompt}

# これまでの会話
{self._format_history(conversation_history)}

# 面接官からの質問
{new_question}

---
{candidate_profile.get("name", "名無しの候補者")} として、最高レベルの志望度と熱意を示しながら、150文字程度で自然な日本語で回答してください。
この企業への強い憧れと、絶対に入社したい気持ちを表現してください。
回答のみを出力し、説明や前置きは不要です。
"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        return messages
