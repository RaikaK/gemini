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
        # 企業情報の"id", "name", "basic_info"キーを削除
        for key in ['id', 'name', 'basic_info']:
            if key in company_profile:
                del company_profile[key]
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
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            response = outputs[0][inputs.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True).strip()
        
        else: # self.model_type == 'api'
            # --- APIモデルでの生成ロジック ---
            return call_openai_api(config.INTERVIEWER_API_MODEL, prompt)

    def ask_question(self, conversation_history):
        """
        まだ話題に上がっていない企業情報の項目について、意図的に質問を生成する。
        """
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history])
        all_company_keys = list(self.company.keys())

        prompt = f"""あなたは、学生の企業研究の深さを測る、戦略的な採用面接官です。
        # あなたが質問できる企業情報の項目リスト
        {all_company_keys}
        # これまでの会話履歴
        {history_str if history_str else "（まだ会話はありません）"}
        # 指示
        1.  **分析**: 上記の「項目リスト」と「会話履歴」を比較し、まだ十分に話題に上がっていない項目は何かを特定してください。
        2.  **質問生成**: 特定した項目の中から、学生の企業理解度を測るために最も効果的なものを1つ選び、それに関する具体的な質問を生成してください。
        思考プロセスや前置きは一切含めず、質問文だけを出力してください。
        質問:"""
        
        question = self._generate_response(prompt, max_tokens=8192)
        thought = "未言及の項目から戦略的に質問を生成しました。"
        return question, thought

    def ask_common_question(self, all_states, all_questions_history):
        """
        全候補者の会話を横断的に分析し、知識の穴を突く戦略的な「全体質問」を生成する。
        """
        conversation_summary = self._format_all_conversations(all_states)
        all_company_keys = list(self.company.keys())
        history_str = "\n".join(f"- {q}" for q in all_questions_history)

        prompt = f"""あなたは、面接全体を俯瞰し、全候補者の理解度を効率的に測る、戦略的な採用面接官です。
        これから全候補者に対して同じ共通質問をします。

        # あなたが質問できる企業情報の項目リスト
        {all_company_keys}

        # これまでの全候補者との会話概要
        {conversation_summary}

        # これまでに行った全ての質問（重複しないように）
        {history_str if history_str else "  (なし)"}

        # 指示
        1.  **全体分析**: 全候補者の会話を俯瞰し、ほとんどの候補者がまだ十分に言及していない「共通の未言及項目」を特定してください。
        2.  **戦略的質問生成**: 特定した項目の中から、候補者たちの企業研究の深さを比較する上で最も重要だと思われるものを1つ選び、それに関する具体的な共通質問を生成してください。

        思考プロセスや前置きは一切含めず、質問文だけを出力してください。
        質問:"""
        
        question = self._generate_response(prompt, max_tokens=8192)
        return question

    def select_least_motivated_candidate(self, all_states):
        """評価タスク1: 最も意欲の低い候補者を選定する"""
        print("--- 最終評価(1/3): 最も意欲の低い候補者の選定を開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        prompt = f"あなたは最終決定権を持つ面接官です。全候補者の記録を確認し、「最も意欲が低い、あるいは準備不足と思われる候補者」を1名だけ選び、名前とその理由を簡潔に述べてください。\n{conversation_summary}\n出力形式:\n最も意欲の低い候補者: [氏名]\n理由: [理由]"
        response = self._generate_response(prompt, max_tokens=8192)
        print("--- 最終評価(1/3): 選定を完了 ---")
        return response

    def rank_candidates_by_motivation(self, all_states):
        """評価タスク2: 候補者を意欲順にランキング付けする"""
        print("--- 最終評価(2/3): 意欲順ランキングを開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        prompt = f"あなたは最終決定権を持つ面接官です。全候補者の記録を確認し、企業への意欲が高い順にランキング付けし、各順位の理由を簡潔に述べてください。\n{conversation_summary}\n出力形式:\n1位: [氏名] (理由: ...)\n2位: [氏名] (理由: ...)\n3位: [氏名] (理由: ...)"
        response = self._generate_response(prompt, max_tokens=8192)
        print("--- 最終評価(2/3): ランキングを完了 ---")
        return response

    def get_all_keys(self, data):
        """辞書を再帰的に走査し、全てのネストされたキーを'parent.child'形式で返す"""
        keys = set()
        def _extract_keys(obj, parent_key=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    keys.add(new_key)
                    _extract_keys(v, new_key)
            elif isinstance(obj, list):
                pass
        _extract_keys(data)
        return keys
    
    def _calculate_detection_metrics(self, llm_output_text, all_states):
        """LLMの出力と正解データを比較し、TP/FP/FNなどの性能メトリクスを計算する"""
        evaluation_results = {}
        candidate_states_map = {s['profile']['name']: s for s in all_states}
        sections = re.split(r'(?=- [^\n]+:)', llm_output_text)
        
        for section in sections:
            section = section.strip()
            if not section or ':' not in section: continue

            first_line = section.split('\n', 1)[0]
            candidate_name = first_line.replace('-', '').strip().split(':', 1)[0].strip()
            if candidate_name not in candidate_states_map: continue
            
            state = candidate_states_map[candidate_name]
            note = None
            detected_missing_keys = set()
            
            key_line_match = re.search(r"欠損項目キー:\s*(\[.*?\])", section)
            if key_line_match:
                try:
                    keys_str = key_line_match.group(1)
                    detected_missing_keys = set(json.loads(keys_str))
                except json.JSONDecodeError:
                    note = "Detected '欠損項目キー' but failed to parse JSON."
            else:
                note = "Candidate block found, but '欠損項目キー' line is missing."

            possessed_knowledge = state['knowledge_tuple'][0]
            actual_missing_keys = {key for key, value in possessed_knowledge.items() if not value}
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
            if note: result["note"] = note
            evaluation_results[candidate_name] = result

        for state in all_states:
            if state['profile']['name'] not in evaluation_results:
                actual_missing_keys = {key for key, value in state['knowledge_tuple'][0].items() if not value}
                evaluation_results[state['profile']['name']] = {
                     "metrics": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "true_positives": 0, "false_positives": 0, "false_negatives": len(actual_missing_keys)},
                    "details": {"correctly_detected_gaps (TP)": [], "incorrectly_detected_gaps (FP)": [], "missed_gaps (FN)": list(actual_missing_keys)},
                    "note": "LLM output for this candidate was not found or failed to parse."}
        return evaluation_results

    def detect_knowledge_gaps(self, all_states):
        """評価タスク3: 知識欠損の定性分析と定量評価を同時に行う"""
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

        llm_analysis_text = self._generate_response(prompt, max_tokens=8192)
        
        performance_metrics = self._calculate_detection_metrics(llm_analysis_text, all_states)
        
        print("--- 最終評価(3/3): 知識欠損の分析と精度評価を完了 ---")
        
        return {
            "llm_qualitative_analysis": llm_analysis_text,
            "quantitative_performance_metrics": performance_metrics
        }
    
    def _format_all_conversations(self, all_states):
        """全候補者の会話ログを整形するヘルパー"""
        full_log = ""
        for i, state in enumerate(all_states):
            profile = state['profile']
            history_str = "\n".join([f"  面接官: {turn['question']}\n  {profile.get('name')}: {turn['answer']}" for turn in state['conversation_log']])
            full_log += f"--- 候補者{i+1}: {profile.get('name')} ---\n会話履歴:\n{history_str}\n\n"
        return full_log.strip()