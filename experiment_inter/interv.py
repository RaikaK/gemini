# interv.py

import torch
import json
import re # 正規表現を扱うためにインポート

class LocalInterviewerLLM:
    """面接官役のLLM (ローカルモデルを使用)"""
    def __init__(self, company_profile, model, tokenizer):
        self.company = company_profile
        self.model = model
        self.tokenizer = tokenizer

    def _generate_response(self, prompt, max_tokens=512):
        """ローカルモデルを使用して応答を生成する共通ヘルパー"""
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
        )
        
        response = outputs[0][inputs.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True).strip()
    
    def ask_common_question(self, already_asked_questions):
        """全候補者向けの共通質問を生成する"""
        asked_questions_str = "\n".join(f"- {q}" for q in already_asked_questions)
        prompt = f"""あなたは、{self.company.get('name')}の採用面接官です。
        これから行う面接で、候補者全員に尋ねるのにふさわしい、ごく標準的な質問を1つだけ考えてください。
        
        # 企業の基本情報
        - 企業名: {self.company.get('name')}
        - 事業内容: {self.company.get('business')}

        # 既に出題した質問 (これらの質問は避けてください)
        {asked_questions_str if asked_questions_str else "（まだ質問はありません）"}
        
        指示:
        - 自己紹介や志望動機、ガクチカ、長所・短所、逆質問など、一般的な質問を生成してください。
        - 次の質問文のみを生成してください。思考プロセスや前置きは一切不要です。
        
        質問:"""
        question = self._generate_response(prompt, max_tokens=100)
        thought = "ローカルモデルが次の全体質問を生成しました。"
        return question, thought

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
            full_log += f"会話履歴:\n{history_str}\n\n"
        return full_log.strip()
        
    def select_best_candidate(self, all_states):
        """評価タスク1: 最も優秀な候補者を1名選出する"""
        print("--- 最終評価(1/3): 最優秀候補者の選定を開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        prompt = f"""あなたは、{self.company.get('name')}の最終決定権を持つ面接官です。
        以下の全候補者の記録を確認し、「最も優秀な候補者」を1名だけ選び、名前と選定理由を100字程度で述べてください。
        
        {conversation_summary}
        
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
        以下の全候補者の記録を確認し、全候補者を評価の高い順にランキング付けし、各順位の理由を簡潔に述べてください。
        
        {conversation_summary}
        
        出力形式:
        1位: [氏名] (理由: ...)
        2位: [氏名] (理由: ...)
        3位: [氏名] (理由: ...)
        """
        response = self._generate_response(prompt, max_tokens=400)
        print("--- 最終評価(2/3): 候補者の順位付けを完了 ---")
        return response

    def _calculate_detection_metrics(self, llm_output_text, all_states):
        """
        LLMの構造化出力と正解データを比較し、検出性能のメトリクスを計算する（堅牢版）。
        """
        evaluation_results = {}

        for state in all_states:
            candidate_name = state['profile']['name']
            note = None
            detected_missing_keys = set()

            # 正規表現で各候補者の分析ブロックを抽出
            # (?s)は複数行にまたがるマッチングを許可するフラグ
            # LLMの出力形式に合わせて、名前の前の "- " を削除
            pattern = re.compile(f"{re.escape(candidate_name)}:(?s)(.*?)(?=\\n\\n|$)")
            match = pattern.search(llm_output_text)
            
            if match:
                candidate_block = match.group(1).strip()
                # 欠損項目キーの行を探す
                key_line_match = re.search(r"欠損項目キー:\s*(\[.*?\])", candidate_block)
                if key_line_match:
                    try:
                        keys_str = key_line_match.group(1)
                        detected_missing_keys = set(json.loads(keys_str))
                    except json.JSONDecodeError:
                        note = "Detected '欠損項目キー' but failed to parse JSON."
                else:
                    note = "Candidate block found, but '欠損項目キー' line is missing."
            else:
                note = "LLM output for this candidate was missing."
            
            # 正解データ（実際に欠損していた項目）
            possessed_knowledge = state['knowledge_tuple'][0]
            actual_missing_keys = {key for key, value in possessed_knowledge.items() if not value}

            # メトリクス計算
            true_positives = actual_missing_keys.intersection(detected_missing_keys)
            false_positives = detected_missing_keys.difference(actual_missing_keys)
            false_negatives = actual_missing_keys.difference(detected_missing_keys)

            tp_count, fp_count, fn_count = len(true_positives), len(false_positives), len(false_negatives)
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            result = {
                "metrics": {
                    "precision": round(precision, 3), "recall": round(recall, 3), "f1_score": round(f1_score, 3),
                    "true_positives": tp_count, "false_positives": fp_count, "false_negatives": fn_count,
                },
                "details": {
                    "correctly_detected_gaps (TP)": list(true_positives),
                    "incorrectly_detected_gaps (FP)": list(false_positives),
                    "missed_gaps (FN)": list(false_negatives),
                }
            }
            if note:
                result["note"] = note
            evaluation_results[candidate_name] = result
            
        return evaluation_results

    def detect_knowledge_gaps(self, all_states):
        """
        評価タスク3: 知識欠損の分析と、その検出精度の評価を同時に行う。
        """
        print("--- 最終評価(3/3): 知識欠損の分析と精度評価を開始 ---")
        
        conversation_summary = self._format_all_conversations(all_states)
        full_company_info_str = json.dumps(self.company, ensure_ascii=False, indent=2)
        
        prompt = f"""あなたは、極めて洞察力の鋭い採用アナリストです。
        以下の「正解の企業情報」と「各候補者の面接記録」を比較し、候補者の知識の穴を特定してください。

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
        1. **思考**: 候補者の各回答を検証します。「この質問に対して、この企業情報（例：'recent_news'）に触れるのが自然だったか？」「回答が具体的か、それとも一般論に終始しているか？」「誤った情報はないか？」といった観点で、知識が欠けていると判断できる「根拠」を探します。
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
        llm_analysis_text = self._generate_response(prompt, max_tokens=512)
        
        performance_metrics = self._calculate_detection_metrics(llm_analysis_text, all_states)
        
        print("--- 最終評価(3/3): 知識欠損の分析と精度評価を完了 ---")
        
        return {
            "llm_qualitative_analysis": llm_analysis_text,
            "quantitative_performance_metrics": performance_metrics
        }