# interv.py

import torch
import json
import re
from utils import call_openai_api
import config

class Interviewer:
    """
    面接官役のLLMを扱う統合クラス。
    ローカルモデルとAPIモデルの両方に対応。
    """
    def __init__(self, company_profile, model_type, model=None, tokenizer=None):
        """
        Args:
            company_profile (dict): 企業情報
            model_type (str): 'local' または 'api'
            model (AutoModelForCausalLM, optional): ローカルモデル. Defaults to None.
            tokenizer (AutoTokenizer, optional): ローカルモデル用トークナイザ. Defaults to None.
        """
        self.company = company_profile
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer

        if self.model_type == 'local' and (not self.model or not self.tokenizer):
            raise ValueError("ローカルモデルタイプには 'model' と 'tokenizer' が必要です。")

    def _generate_response(self, prompt, max_tokens=512):
        """モデルタイプに応じて応答を生成する"""
        if self.model_type == 'local':
            # --- ローカルモデルでの生成ロジック ---
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
                do_sample=True, temperature=0.6, top_p=0.9,
            )
            response = outputs[0][inputs.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True).strip()

        elif self.model_type == 'api':
            # --- APIモデルでの生成ロジック ---
            system_prompt = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
            full_prompt = f"システム指示: {system_prompt}\n\nユーザー指示:\n{prompt}"
            return call_openai_api(config.INTERVIEWER_API_MODEL, full_prompt)
        
        else:
            raise ValueError(f"無効なモデルタイプです: {self.model_type}")

    def ask_common_question(self, already_asked_questions):
        """全候補者向けの共通質問を生成する"""
        asked_questions_str = "\n".join(f"- {q}" for q in already_asked_questions)
        prompt = f"""あなたは、{self.company.get('name')}の採用面接官です。
        候補者全員に尋ねるのにふさわしい、ごく標準的な質問を1つだけ考えてください。
        # 既に出題した質問 (これらの質問は避けてください)
        {asked_questions_str if asked_questions_str else "（まだ質問はありません）"}
        指示: 次の質問文のみを生成してください。思考プロセスや前置きは一切不要です。
        質問:"""
        question = self._generate_response(prompt, max_tokens=100)
        thought = f"{self.model_type}モデルが次の全体質問を生成しました。"
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
        thought = f"{self.model_type}モデルが次の質問を生成しました。"
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

    def select_least_motivated_candidate(self, all_states):
        """評価タスク1: 最も志望度が低い候補者を1名選出する"""
        print(f"--- 最終評価(1/3): 最も志望度が低い候補者の選定を開始 ({self.model_type}モデル) ---")
        conversation_summary = self._format_all_conversations(all_states)
        prompt = f"""あなたは、{self.company.get('name')}の最終決定権を持つ面接官です。
        以下の全候補者の記録を確認し、「最も志望度が低いと思われる候補者」を1名だけ選び、名前と選定理由を100字程度で述べてください。
        {conversation_summary}
        出力形式:
        最も志望度が低い候補者: [氏名]
        選定理由: [理由]
        """
        response = self._generate_response(prompt, max_tokens=200)
        print("--- 最終評価(1/3): 人狼の選定を完了 ---")
        return response

    def rank_candidates_by_motivation(self, all_states):
        """評価タスク2: 全候補者を志望度が低い順にランキング付けする"""
        print(f"--- 最終評価(2/3): 候補者の志望度順位付けを開始 ({self.model_type}モデル) ---")
        conversation_summary = self._format_all_conversations(all_states)
        prompt = f"""あなたは、{self.company.get('name')}の最終決定権を持つ面接官です。
        以下の全候補者の記録を確認し、全候補者を志望度が低いと思われる順にランキング付けし、各順位の理由を簡潔に述べてください。
        
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
            pattern = re.compile(f"{re.escape(candidate_name)}:(.*?)(?=\\n\\n|$)", re.DOTALL)
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
                "metrics": {"precision": round(precision, 3), "recall": round(recall, 3), "f1_score": round(f1_score, 3)},
                "details": {"correctly_detected (TP)": list(true_positives), "incorrectly_detected (FP)": list(false_positives), "missed (FN)": list(false_negatives)}
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
        llm_analysis_text = self._generate_response(prompt, max_tokens=1024) # 少し長めに変更
        performance_metrics = self._calculate_detection_metrics(llm_analysis_text, all_states)
        
        print("--- 最終評価(3/3): 知識欠損の分析と精度評価を完了 ---")
        
        return {
            "llm_qualitative_analysis": llm_analysis_text,
            "quantitative_performance_metrics": performance_metrics
        }