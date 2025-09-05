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
        thought = f"{self.model_type}モデルが次の戦略的全体質問を生成しました。"
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

    def should_continue_interview(self, conversation_history, current_round, max_rounds=5):
        """面接を続けるべきかどうかを判断する"""
        if current_round >= max_rounds:
            return False, "最大ラウンド数に達しました"
        
        if not conversation_history:
            return True, "初回のため継続"
        
        # 最近の回答を分析して自信度を判定
        recent_answers = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent_answers])
        
        prompt = f"""あなたは、{self.company.get('name')}の採用面接官です。
        現在の面接ラウンド: {current_round}/{max_rounds}
        
        以下の最近の会話履歴を分析し、候補者の志望度について十分な情報が得られているか判断してください。
        
        会話履歴:
        {history_str}
        
        判断基準:
        1. 候補者の志望度について明確な判断ができるか
        2. 他の候補者との比較に十分な情報があるか
        3. 面接を続けることで新たな洞察が得られる可能性があるか
        
        回答形式:
        - 継続する場合: "CONTINUE"
        - 終了する場合: "STOP"
        - 理由: [簡潔な理由]
        
        回答:"""
        
        response = self._generate_response(prompt, max_tokens=200)
        
        # レスポンスを解析
        if "CONTINUE" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "追加情報が必要"
            return True, reason
        elif "STOP" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "十分な情報が得られた"
            return False, reason
        else:
            # デフォルトは継続
            return True, "判断できないため継続"

    def decide_next_question_type(self, candidate_states, asked_common_questions, current_round, max_rounds=5):
        """次の質問タイプ（全体質問 vs 個別質問）を智的に決定する"""
        
        # 基本的な条件チェック
        if current_round == 1:
            return "common", "初回は全体質問から開始"
        
        if len(asked_common_questions) == 0:
            return "common", "まだ全体質問を行っていない"
        
        # 全候補者の回答状況を分析
        all_responses_count = sum(len(state['conversation_log']) for state in candidate_states)
        avg_responses_per_candidate = all_responses_count / len(candidate_states) if candidate_states else 0
        
        # 最新の全体質問からの経過ラウンド数を計算
        last_common_round = 0
        for state in candidate_states:
            for turn in state['conversation_log']:
                # 全候補者が同じ質問を受けているかチェック（全体質問の特徴）
                if turn['turn'] > last_common_round:
                    # 他の候補者も同じ質問を受けているかチェック
                    same_question_count = sum(1 for other_state in candidate_states 
                                            if any(other_turn['question'] == turn['question'] 
                                                  for other_turn in other_state['conversation_log']))
                    if same_question_count == len(candidate_states):
                        last_common_round = turn['turn']
        
        rounds_since_common = current_round - last_common_round
        
        # LLMによる状況分析
        situation_summary = self._analyze_interview_situation(candidate_states, asked_common_questions, current_round)
        
        prompt = f"""あなたは、{self.company.get('name')}の面接戦略エキスパートです。
        現在の面接状況を分析し、次に行うべき質問タイプを決定してください。

        # 現在の状況
        - 現在のラウンド: {current_round}/{max_rounds}
        - 実施済み全体質問数: {len(asked_common_questions)}
        - 候補者あたり平均回答数: {avg_responses_per_candidate:.1f}
        - 最後の全体質問からの経過ラウンド: {rounds_since_common}
        
        # 状況分析
        {situation_summary}
        
        # 判断基準
        【全体質問を選ぶべき場合】
        - 候補者間の比較材料が不足している
        - 特定の重要なトピックについて全員の見解が必要
        - 個別質問で深掘りする前に基盤となる情報が必要
        - 最後の全体質問から時間が経ちすぎている（3ラウンド以上）
        
        【個別質問を選ぶべき場合】
        - 特定の候補者の回答をより深く探る必要がある
        - 候補者ごとに異なる角度からの質問が効果的
        - 志望度の判定に必要な個人的な動機を探る必要がある
        - 十分な全体質問が既に実施されている
        
        回答形式:
        決定: COMMON または INDIVIDUAL
        理由: [詳細な理由を100字程度で]
        
        回答:"""
        
        response = self._generate_response(prompt, max_tokens=300)
        
        # レスポンスを解析
        if "COMMON" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "全体質問が適切"
            return "common", reason
        elif "INDIVIDUAL" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "個別質問が適切"
            return "individual", reason
        else:
            # デフォルトの判断ロジック
            if rounds_since_common >= 3 or len(asked_common_questions) < 2:
                return "common", "バランスを保つため全体質問を選択"
            else:
                return "individual", "深掘りのため個別質問を選択"

    def _analyze_interview_situation(self, candidate_states, asked_common_questions, current_round):
        """現在の面接状況を分析して要約を作成"""
        
        # 各候補者の回答の特徴を分析
        candidate_analysis = []
        for i, state in enumerate(candidate_states):
            name = state['profile'].get('name', f'候補者{i+1}')
            response_count = len(state['conversation_log'])
            
            # 最新の回答の長さと内容の傾向を分析
            if state['conversation_log']:
                recent_answers = [turn['answer'] for turn in state['conversation_log'][-2:]]
                avg_answer_length = sum(len(answer) for answer in recent_answers) / len(recent_answers)
                
                # 回答の詳細度を簡易評価
                detail_level = "高い" if avg_answer_length > 100 else "中程度" if avg_answer_length > 50 else "低い"
                candidate_analysis.append(f"- {name}: 回答数{response_count}, 詳細度{detail_level}")
            else:
                candidate_analysis.append(f"- {name}: 未回答")
        
        # 質問のバランス分析
        common_questions_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(asked_common_questions))
        
        situation = f"""
候補者別の回答状況:
{chr(10).join(candidate_analysis)}

実施済み全体質問:
{common_questions_str if common_questions_str else "  なし"}

現在の課題:
- 候補者間の比較材料の充実度
- 各候補者の志望度判定に必要な情報の蓄積状況
- 面接の残り時間と効率性のバランス
        """
        
        return situation.strip()

    def conduct_dynamic_interview(self, candidate_states, applicant, max_rounds=5):
        """智的な動的面接フローを実行する"""
        print(f"--- 智的動的面接フロー開始 (最大{max_rounds}ラウンド) ---")
        
        asked_common_questions = []
        current_round = 0
        actual_interview_flow = []  # 実際に実行された面接フローを記録
        
        while current_round < max_rounds:
            current_round += 1
            print(f"--- 面接ラウンド {current_round}/{max_rounds} ---")
            
            # 次の質問タイプを智的に決定
            question_type, reason = self.decide_next_question_type(
                candidate_states, asked_common_questions, current_round, max_rounds
            )
            
            print(f"--- 選択された質問タイプ: {question_type} ---")
            print(f"--- 選択理由: {reason} ---")
            
            if question_type == "common":
                # 全体質問フェーズ
                print("--- 全体質問フェーズを実行 ---")
                actual_interview_flow.append(0)  # 0 = 全体質問
                question, _ = self.ask_common_question(candidate_states, asked_common_questions)
                asked_common_questions.append(question)
                print(f"--- 生成された全体質問: 「{question}」 ---")
                
                for i, state in enumerate(candidate_states):
                    print(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                    # 学生の回答を生成
                    answer = applicant.generate(
                        state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                    )
                    print(f"学生 (API): {answer}")
                    state["conversation_log"].append({"turn": current_round, "question": question, "answer": answer})
                
                # 全体質問後の継続判断
                overall_responses = [state['conversation_log'][-1] for state in candidate_states if state['conversation_log']]
                should_continue, continue_reason = self.should_continue_interview(
                    overall_responses, current_round, max_rounds
                )
                print(f"全体質問後の継続判断: {'継続' if should_continue else '終了'} - {continue_reason}")
                if not should_continue:
                    break
                    
            elif question_type == "individual":
                # 個別質問フェーズ
                print("--- 個別質問フェーズを実行 ---")
                actual_interview_flow.append(1)  # 1 = 個別質問
                any_continued = False
                
                for i, state in enumerate(candidate_states):
                    print(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} への個別面接判断")
                    
                    # 個別質問の継続判断
                    should_continue, reason = self.should_continue_interview(
                        state['conversation_log'], current_round, max_rounds
                    )
                    
                    if should_continue:
                        any_continued = True
                        question, _ = self.ask_question(state['conversation_log'])
                        print(f"面接官 ({self.model_type}): {question}")
                        # 学生の回答を生成
                        answer = applicant.generate(
                            state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                        )
                        print(f"学生 (API): {answer}")
                        state["conversation_log"].append({"turn": current_round, "question": question, "answer": answer})
                    else:
                        print(f"候補者 {i+1} の個別面接終了: {reason}")
                
                # 誰も継続しない場合は面接終了
                if not any_continued:
                    print("--- 全候補者の個別面接が完了したため面接終了 ---")
                    break
            
            # 面接全体の進捗評価
            if current_round >= 3:  # 最低3ラウンド後に全体評価
                overall_should_continue, overall_reason = self._evaluate_overall_progress(
                    candidate_states, current_round, max_rounds
                )
                print(f"面接全体の進捗評価: {'継続' if overall_should_continue else '終了'} - {overall_reason}")
                if not overall_should_continue:
                    break
        
        print(f"--- 智的動的面接フロー完了 (実行ラウンド数: {current_round}) ---")
        print(f"--- 実際の面接フロー: {actual_interview_flow} ---")
        return current_round, actual_interview_flow

    def _evaluate_overall_progress(self, candidate_states, current_round, max_rounds):
        """面接全体の進捗を評価し、継続の必要性を判断"""
        
        # 各候補者の情報収集状況を分析
        candidate_info_summary = []
        for i, state in enumerate(candidate_states):
            name = state['profile'].get('name', f'候補者{i+1}')
            response_count = len(state['conversation_log'])
            
            if state['conversation_log']:
                total_answer_length = sum(len(turn['answer']) for turn in state['conversation_log'])
                avg_answer_length = total_answer_length / response_count
                info_richness = "豊富" if avg_answer_length > 120 else "標準" if avg_answer_length > 60 else "限定的"
            else:
                info_richness = "なし"
            
            candidate_info_summary.append(f"- {name}: {response_count}回答, 情報量{info_richness}")
        
        summary_text = "\n".join(candidate_info_summary)
        
        prompt = f"""あなたは、{self.company.get('name')}の面接効率化エキスパートです。
        現在の面接の進捗状況を評価し、志望度の低い候補者を特定するのに十分な情報が得られているか判断してください。

        # 現在の状況
        - 現在のラウンド: {current_round}/{max_rounds}
        - 各候補者の情報収集状況:
        {summary_text}

        # 判断基準
        1. 各候補者の志望度を判断するのに十分な回答が得られているか
        2. 候補者間の比較が可能な材料が揃っているか
        3. 残りのラウンドで得られる追加情報の価値
        4. 面接の効率性（時間対効果）

        # 終了条件
        - 全候補者から志望度判定に必要な情報が得られた
        - 候補者間の差が明確になった
        - 追加の質問をしても新たな洞察が得られそうにない

        回答形式:
        判定: CONTINUE または STOP
        理由: [判断の根拠を100字程度で]

        回答:"""
        
        response = self._generate_response(prompt, max_tokens=300)
        
        # レスポンスを解析
        if "CONTINUE" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "さらなる情報収集が必要"
            return True, reason
        elif "STOP" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "十分な情報が収集された"
            return False, reason
        else:
            # デフォルトは継続（安全側）
            return True, "判断不明のため継続"

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