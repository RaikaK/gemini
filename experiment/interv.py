# 面接官役のLLMを定義するファイルです。
# Gemini APIを使用して、面接官の質問生成や学生評価を行います


import torch
import json
from utils import call_gemini_api, parse_json_from_response
from student import InstructionPromptManager
from config import INTERVIEWER_MODEL_NAME

class InterviewerLLM:
    """面接官役のLLM (Gemini APIを使用)"""
    def __init__(self, company_profile):
        self.model_name = INTERVIEWER_MODEL_NAME
        self.company = company_profile

    def ask_question(self, conversation_history):
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history])
        prompt = f"""あなたは、{self.company.get('name', '私たちの会社')} という企業の採用面接官です。企業の詳細: {self.company.get('business', 'N/A')}
        以下の会話履歴を読み、学生の回答を分析し、次にするべき質問を考えてください。学生の企業理解度、論理性、熱意を測ることを目的とします。
        思考プロセスを内部で実行し、最終的な「質問」だけを返してください。
        ---
        会話履歴:
        {history_str if history_str else "（まだ会話はありません）"}
        ---
        次の質問:"""
        question = call_gemini_api(self.model_name, prompt)
        thought = f"履歴を分析し、質問 '{question[:30]}...' を生成しました。"
        return question, thought

    def evaluate_applicant(self, conversation_history, applicant_profile):
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history])
        prompt = f"""あなたは、{self.company.get('name', '私たちの会社')} のベテラン面接官です。以下の学生プロフィールと面接の会話履歴全体をレビューし、最終評価を行ってください。
        [企業情報]: {json.dumps(self.company, ensure_ascii=False)}
        [学生プロフィール]: {json.dumps(applicant_profile, ensure_ascii=False)}
        [面接の全会話履歴]:\n{history_str}
        [評価項目]: 一貫性, 具体性, 論理性, 熱意 (各1-5点)
        [出力形式]: 以下のキーを持つJSON形式で評価を出力してください。
        {{
          "overall_score": 総合評価 (5段階評価の浮動小数点数),
          "summary": "評価の要約 (100字程度)",
          "evaluation_details": {{"consistency":点数,"specificity":点数,"logicality":点数,"enthusiasm":点数}}
        }}"""
        response = call_gemini_api(self.model_name, prompt)
        return parse_json_from_response(response)

class LLamaInterviewResponseGenerator:
    """ローカルLLMからの応答生成を統括するクラス"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_manager = InstructionPromptManager()

    def generate(self, candidate_profile, company_knowledge_tuple, conversation_history, new_question):
        messages = self.prompt_manager.create_messages(
            candidate_profile, company_knowledge_tuple, conversation_history, new_question
        )
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        attention_mask = torch.ones_like(inputs).to(self.model.device)
        inputs = inputs.to(self.model.device)
        
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        response = outputs[0][inputs.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
