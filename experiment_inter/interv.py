# interv.py

import torch
import json

class LocalInterviewerLLM:
    """面接官役のLLM (ローカルモデルを使用)"""
    def __init__(self, company_profile, model, tokenizer):
        self.company = company_profile
        self.model = model
        self.tokenizer = tokenizer

    def _generate_response(self, prompt, max_tokens=256):
        """ローカルモデルを使用して応答を生成する共通ヘルパー"""
        messages = [
            {"role": "system", "content": "あなたは公平で鋭い質問をする採用面接官です。与えられた指示に日本語で正確に従ってください。"},
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
        )
        
        response = outputs[0][inputs.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True).strip()

    def ask_question(self, conversation_history):
        """会話履歴に基づき、次の質問を生成する"""
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history])
        prompt = f"""あなたは、{self.company.get('name')}の採用面接官です。
        以下の会話履歴を読み、学生の回答を深掘りするための次の質問を1つだけ考えてください。
        
        会話履歴:
        {history_str if history_str else "（まだ会話はありません）"}
        
        指示: 次の質問のみを生成してください。思考プロセスや前置きは不要です。
        質問:"""
        question = self._generate_response(prompt, max_tokens=100)
        thought = "ローカルモデルが次の質問を生成しました。"
        return question, thought

    def _format_all_conversations(self, all_states):
        """最終評価のために、全候補者の会話ログを整形するヘルパー"""
        full_log = ""
        for i, state in enumerate(all_states):
            profile = state['profile']
            history_str = "\n".join([f"  面接官: {turn['question']}\n  {profile.get('name')}: {turn['answer']}" for turn in state['conversation_log']])
            full_log += f"--- 候補者{i+1}: {profile.get('name')} ---\n"
            full_log += f"プロフィール: {json.dumps(profile, ensure_ascii=False)}\n"
            full_log += f"会話履歴:\n{history_str}\n\n"
        return full_log.strip()
        
    def select_best_candidate(self, all_states):
        """評価タスク1: 最も優秀な候補者を1名選出する"""
        print("--- 最終評価(1/3): 最優秀候補者の選定を開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        prompt = f"""あなたは、{self.company.get('name')}の最終決定権を持つ面接官です。
        以下の全候補者の記録を確認してください。
        
        {conversation_summary}
        
        指示: 上記候補者から「最も優秀な候補者」を1名だけ選び、名前と選定理由を100字程度で述べてください。
        
        出力形式:
        最優秀候補者: [氏名]
        選定理由: [理由]
        """
        response = self._generate_response(prompt, max_tokens=200)
        print("--- 最終評価(1/3): 最優秀候補者の選定を完了 ---")
        return response

    def rank_candidates(self, all_states):
        """評価タスク2: 全候補者をランキング付けする"""
        print("--- 最終評価(2/3): 候補者の順位付けを開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        prompt = f"""あなたは、{self.company.get('name')}の最終決定権を持つ面接官です。
        以下の全候補者の記録を確認してください。
        
        {conversation_summary}
        
        指示: 全候補者を評価の高い順に1位からランキング付けし、各順位の理由を50字程度で簡潔に述べてください。
        
        出力形式:
        1位: [氏名] (理由: ...)
        2位: [氏名] (理由: ...)
        3位: [氏名] (理由: ...)
        """
        response = self._generate_response(prompt, max_tokens=400)
        print("--- 最終評価(2/3): 候補者の順位付けを完了 ---")
        return response

    def detect_knowledge_gaps(self, all_states):
        """評価タスク3: 各候補者の知識の穴を特定する"""
        print("--- 最終評価(3/3): 知識欠損の分析を開始 ---")

        conversation_summary = self._format_all_conversations(all_states)

        prompt = f"""
        あなたは、{self.company.get('name')}の採用アナリストです。
        以下の全候補者の面接会話ログと、当社の正しい企業情報を厳密に比較し、各候補者が回答内で言及した情報のうち、正しい企業情報と異なっている項目をすべて特定してください。

        ### 当社の正しい企業情報:
        {self.company}

        ### 候補者との会話ログ:
        {conversation_summary}

        ### 指示:
        各候補者について、以下のフォーマットで分析結果を出力してください。
        1.  **回答内で言及された企業情報項目を正確に抽出する。**
        2.  **抽出した情報が正しい情報と一致するかを判断する。**
        3.  **不一致（誤り）または言及が全くない項目を「知識欠損項目」としてリストアップする。**
        4.  **なぜその項目が欠損していると判断したのか、その理由を具体的に記述する。**

        ### 出力形式:
        - {all_states[0]['profile']['name']}:
        - 知識欠損項目:
            - [項目名]: [なぜ欠損していると判断したのかの理由]
        ...

        - {all_states[1]['profile']['name']}:
        - 知識欠損項目:
            - [項目名]: [なぜ欠損していると判断したのかの理由]
        ...

        - {all_states[2]['profile']['name']}:
        - 知識欠損項目:
            - [項目名]: [なぜ欠損していると判断したのかの理由]
        ...

        """
        response = self._generate_response(prompt, max_tokens=500)
        print("--- 最終評価(3/3): 知識欠損の分析を完了 ---")
        return response