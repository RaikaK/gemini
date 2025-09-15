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

        elif self.model_type == 'api':
            # --- APIモデルでの生成ロジック ---
            system_prompt = "あなたは与えられた指示に日本語で正確に従う、非常に優秀で洞察力のある採用アナリストです。"
            full_prompt = f"システム指示: {system_prompt}\n\nユーザー指示:\n{prompt}"
            response_text, _ = call_openai_api(config.INTERVIEWER_API_MODEL, full_prompt)
            return response_text
        
        else:
            raise ValueError(f"無効なモデルタイプです: {self.model_type}")

    def ask_common_question(self, all_questions_history):
        """
        全候補者の会話を横断的に分析し、知識の穴を突く戦略的な「全体質問」を生成する。
        """
        all_company_keys = list(self.company.keys())
        history_str = "\n".join(f"- {q}" for q in all_questions_history)

        prompt = f"""あなたは、面接全体を俯瞰し、全候補者の理解度を効率的に測る、戦略的な採用面接官です。
        これから全候補者に対して同じ共通質問をします。

        # あなたが質問できる企業情報の項目リスト
        {all_company_keys}

        # これまでに行った全ての質問（重複しないように）
        {history_str if history_str else "  (なし)"}

        【全体質問の戦略的役割】
        全体質問は以下の目的で使用されます：
        1. **比較基準の確立**: 全候補者が同じ条件で回答するため、公平な比較が可能
        2. **基盤情報の収集**: 候補者間の差別化に必要な基本的な企業理解度を測定
        3. **効率的な情報収集**: 一度の質問で全候補者から情報を収集し、時間を節約
        4. **共通トピックの深掘り**: 特定の重要な企業情報について全員の見解を比較

        【全体質問を選ぶべき戦略的状況】
        - 候補者間の比較材料が不足している場合
        - 特定の重要な企業情報について全員の理解度を測りたい場合
        - 個別質問で深掘りする前に基盤となる情報が必要な場合
        - 効率的に情報収集を進めたい場合

        # 指示
        1.  **全体分析**: 全候補者の会話を俯瞰し、ほとんどの候補者がまだ十分に言及していない「共通の未言及項目」を特定してください。
        2.  **戦略的質問生成**: 特定した項目の中から、候補者たちの企業研究の深さを比較する上で最も重要だと思われるものを1つ選び、それに関する具体的な共通質問を生成してください。
        3.  **比較可能性の確保**: 全候補者が同じ基準で回答できる質問であることを確認してください。

        思考プロセスや前置きは一切含めず、質問文だけを出力してください。
        質問:"""
        
        question = self._generate_response(prompt, max_tokens=8192)
        thought = f"{self.model_type}モデルが次の戦略的全体質問を生成しました。"
        return question, thought

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

        【個別質問の戦略的役割】
        個別質問は以下の目的で使用されます：
        1. **深掘り調査**: 特定の候補者の回答をより深く探り、詳細な情報を収集
        2. **個人的動機の探求**: 候補者固有の志望動機や背景を理解
        3. **知識欠損の特定**: この候補者が特に不足している企業知識を特定
        4. **差別化要因の発見**: 他の候補者との違いを明確にする情報を収集
        5. **曖昧な回答の明確化**: 以前の回答で不明確だった部分を明確にする

        【個別質問を選ぶべき戦略的状況】
        - 特定の候補者の回答をより深く探る必要がある場合
        - 候補者ごとに異なる角度からの質問が効果的な場合
        - 志望度の判定に必要な個人的な動機を探る必要がある場合
        - 十分な全体質問が既に実施されている場合
        - 情報が不足している候補者が特定されている場合

        # 指示
        1.  **分析**: 上記の「項目リスト」と「会話履歴」を比較し、まだ十分に話題に上がっていない項目は何かを特定してください。
        2.  **戦略的質問生成**: 特定した項目の中から、この学生の企業理解度を測るために最も効果的なものを1つ選び、それに関する具体的な質問を生成してください。
        3.  **個別性の確保**: この候補者に特化した、深掘りできる質問であることを確認してください。
        4.  **知識欠損の特定**: この候補者が特に不足している可能性が高い企業知識に焦点を当ててください。

        思考プロセスや前置きは一切含めず、質問文だけを出力してください。
        質問:"""
        
        question = self._generate_response(prompt, max_tokens=8192)
        thought = "未言及の項目から戦略的に質問を生成しました。"
        return question, thought
    
    def _select_candidate_with_llm(self, candidate_states):
            """LLMに候補者会話ログを渡して、最も理解が浅い候補者を選ばせる"""
            conversation_summary = self._format_all_conversations(candidate_states)

            prompt = f"""あなたは採用面接官です。
        以下に複数候補者の会話履歴があります。誰が最も企業理解が浅いかを判断してください。

        # 候補者の会話ログ
        {conversation_summary}

        出力形式は必ず次のJSONで返してください:
        {{
          "weakest_candidate_index": 数値,  # 0始まりのインデックス
          "reason": "簡潔な理由"
        }}"""

            response = self._generate_response(prompt, max_tokens=800)
            try:
                data = json.loads(response)
                return data.get("weakest_candidate_index", 0), data.get("reason", "")
            except Exception:
                # JSONパース失敗時は0番候補者にフォールバック
                return 0, "LLM出力を解釈できなかったためフォールバック"

    # interv.py の Interviewer クラスに追加
    def run_interview_round(self, question_type, candidate_states, applicant, 
                            asked_common_questions, current_round, log_fn=print):
        """
        1ラウンド分の面接を実施する共通処理
        Args:
            question_type: 0 = 全体質問, 1 = 個別質問
            candidate_states: 各候補者の状態リスト
            applicant: 学生モデル
            asked_common_questions: 既に出た全体質問リスト
            current_round: 現在のラウンド番号
            log_fn: ログ出力用関数 (デフォルトprint、app.pyではlog_messageを渡す)
        Returns:
            current_round: 更新後のラウンド番号
        """
        if question_type == 0:
            # 全体質問
            # current_round += 1
            log_fn(f"--- 面接ラウンド {current_round} (全体質問) ---")
            question, _ = self.ask_common_question(asked_common_questions)
            asked_common_questions.append(question)
            log_fn(f"--- 生成された全体質問: 「{question}」 ---")

            for i, state in enumerate(candidate_states):
                log_fn(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                answer, token_info = applicant.generate(
                    state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                )
                log_fn(f"学生: {answer}")
                state["conversation_log"].append({
                    "turn": current_round, "question": question,
                    "answer": answer, "token_info": token_info
                })

        elif question_type == 1:
            # 個別質問（ランダム1人のみ）
            target_index, reason = self._select_candidate_with_llm(candidate_states)
            state = candidate_states[target_index]
            # current_round += 1
            log_fn(f"--- 面接ラウンド {current_round} (個別質問 - 候補者 {target_index+1}) ---")
            question, _ = self.ask_question(state["conversation_log"])
            log_fn(f"面接官: {question}")
            answer, token_info = applicant.generate(
                state["profile"], state["knowledge_tuple"], state["conversation_log"], question
            )
            log_fn(f"学生: {answer}")
            state["conversation_log"].append({
                "turn": current_round, "question": question,
                "answer": answer, "token_info": token_info
            })

        return current_round


    def conduct_dynamic_interview(self, candidate_states, applicant, max_rounds=10):
        """智的な動的面接フローを実行する（個別質問はランダムに1人だけ対象）"""
        print(f"--- 智的動的面接フロー開始 (最大{max_rounds}ラウンド) ---")

        asked_common_questions = []
        current_round = 0
        actual_interview_flow = []  # 実際に実行された面接フローを記録

        # 各候補者の個別質問回数を追跡（元の形式を維持：dict で管理）
        individual_question_counts = {i: 0 for i in range(len(candidate_states))}

        while current_round < max_rounds:
            current_round += 1
            print(f"--- 面接ラウンド {current_round}/{max_rounds} ---")

            question_type, reason = self.decide_next_question_type_balanced(
                candidate_states, asked_common_questions, current_round, max_rounds, individual_question_counts
            )

            print(f"--- 選択された質問タイプ: {question_type} ---")
            print(f"--- 選択理由: {reason} ---")

            if question_type == "common":
                current_round = self.run_interview_round(
                    0, candidate_states, applicant, asked_common_questions, current_round, log_fn=print
                )
                actual_interview_flow.append(0)

            elif question_type == "individual":
                current_round = self.run_interview_round(
                    1, candidate_states, applicant, asked_common_questions, current_round, log_fn=print
                )
                # ※ run_interview_round 内で誰に聞いたかは元の実装に従います
                #   もし個別ターゲットのカウントを正確に取りたい場合は、
                #   run_interview_round 側で対象indexを返すようにする必要あり
                actual_interview_flow.append(1)

            # 面接全体の進捗評価（最低3ラウンド後に実施）
            if current_round >= 3:
                overall_should_continue, overall_reason = self._evaluate_overall_progress(
                    candidate_states, current_round, max_rounds
                )
                print(f"面接全体の進捗評価: {'継続' if overall_should_continue else '終了'} - {overall_reason}")
                if not overall_should_continue:
                    break
                
        print(f"--- 智的動的面接フロー完了 (実行ラウンド数: {current_round}) ---")
        print(f"--- 実際の面接フロー: {actual_interview_flow} ---")
        print(f"--- 各候補者の個別質問回数: {individual_question_counts} ---")
        return current_round, actual_interview_flow

    def _evaluate_overall_progress(self, candidate_states, current_round, max_rounds, threshold: float = 0.3):
        """
        面接全体の進捗を評価し、継続の必要性を判断する（定量基準版）

        - 各候補者について未質問キーを検出
        - 未質問率 = 未質問キー数 / 企業キー総数
        - 全候補者の未質問率の最小値が threshold 未満なら STOP
        """
        total_keys = len(self.company.keys())
        candidate_info_summary = []
        max_unasked_ratio = 1.0

        for i, state in enumerate(candidate_states):
            name = state['profile'].get('name', f'候補者{i+1}')

            # 未質問キーを検出
            missing_keys = self._detect_unasked_keys(state["conversation_log"], self.company)
            unasked_ratio = len(missing_keys) / total_keys if total_keys > 0 else 0.0

            max_unasked_ratio = max(max_unasked_ratio, unasked_ratio)
            candidate_info_summary.append(f"- {name}: 未質問率 {unasked_ratio:.2f} ({len(missing_keys)}/{total_keys})")

        summary_text = "\n".join(candidate_info_summary)

        # 判定ロジック
        if max_unasked_ratio <= threshold:
            reason = f"最小未質問率 {max_unasked_ratio:.2f} が閾値 {threshold} を下回ったため、十分な情報が収集されたと判断"
            return False, reason + "\n" + summary_text
        else:
            reason = f"最小未質問率 {max_unasked_ratio:.2f} が閾値 {threshold} を上回っているため、さらなる質問が必要"
            return True, reason + "\n" + summary_text

    def decide_next_question_type_balanced(
        self,
        candidate_states,
        asked_common_questions,
        current_round,
        max_rounds,
        individual_question_counts
    ):
        """
        バランス指標に基づき、次の質問タイプを決定する
        - 全体質問スコア: 最近全体質問が少ない/比較材料が不足している
        - 個別質問スコア: まだ質問をしていない項目数/質問回数の偏り
        - ラウンド進行度で重み調整（序盤=全体寄り, 終盤=個別寄り）
        戻り値: ( "common" or "individual", 理由文字列 )
        """
        # --- 前処理 ---
        num_candidates = len(candidate_states)
        total_company_keys = len(self.company.keys()) if hasattr(self, "company") else 0
        total_company_keys = max(1, total_company_keys)  # 0割防止

        # 進行度 (0.0=序盤, 1.0=終盤)
        progress = current_round / max_rounds if max_rounds > 0 else 1.0

        # -------------------------
        # 欠損度（候補者ごと）
        # -------------------------
        deficiency_scores = []  # 各候補者: 欠損キー数/全キー数
        for state in candidate_states:
            missing_keys = self._detect_missing_keys(state["conversation_log"], self.company)
            deficiency_scores.append(len(missing_keys) / total_company_keys)

        max_deficiency = max(deficiency_scores) if deficiency_scores else 0.0

        # -------------------------
        # 全体質問スコア
        # -------------------------
        common_score = 0.5

        # (a) 最近、全体質問が少なければスコア↑
        #     直近の頻度を見る代替として「ここまでに実施した共通質問の割合」を使用
        prev_rounds = max(1, current_round - 1)
        common_ratio_so_far = len(asked_common_questions) / prev_rounds
        if len(asked_common_questions) == 0:
            common_score += 0.25  # まだ一度も共通質問していない → 強めにブースト
        elif common_ratio_so_far < 0.3:
            common_score += 0.15  # 共通質問の比率が低い

        # (b) 比較材料不足（全員の回答ターン数がほぼ同じ）ならスコア↑
        answer_counts = [len(state["conversation_log"]) for state in candidate_states] or [0]
        if len(set(answer_counts)) <= 1:
            common_score += 0.2

        # -------------------------
        # 個別質問スコア
        # -------------------------
        individual_score = 0.5

        # (a) 欠損度が高い候補者がいるほどスコア↑
        individual_score += 0.6 * max_deficiency  # 欠損度をやや強めに反映

        # (b) フェアネス（個別質問回数の少ない候補者がいる場合）でスコア↑
        if isinstance(individual_question_counts, dict) and individual_question_counts:
            counts = list(individual_question_counts.values())
            min_q = min(counts)
            max_q = max(counts)
            if min_q == 0:
                individual_score += 0.15  # まだ個別を当てられていない人がいる
            elif (max_q - min_q) >= 2:
                individual_score += 0.10  # 偏りが大きい

        # -------------------------
        # 進行度による重み付け
        # -------------------------
        weighted_common = common_score * (1 - progress)
        weighted_individual = individual_score * progress

        # -------------------------
        # 判定
        # -------------------------
        if weighted_common >= weighted_individual:
            reason = (
                f"common選択: progress={progress:.2f}, common_score={common_score:.2f}, "
                f"individual_score={individual_score:.2f}, max_def={max_deficiency:.2f} "
                f"(共通比率={common_ratio_so_far:.2f}, 比較材料不足={'Yes' if len(set(answer_counts))<=1 else 'No'})"
            )
            return "common", reason
        else:
            reason = (
                f"individual選択: progress={progress:.2f}, common_score={common_score:.2f}, "
                f"individual_score={individual_score:.2f}, max_def={max_deficiency:.2f} "
                f"(共通比率={common_ratio_so_far:.2f}, 比較材料不足={'Yes' if len(set(answer_counts))<=1 else 'No'})"
            )
            return "individual", reason
        

    def _detect_unasked_keys(self, conversation_log, company_profile):
        """
        LLMを使って候補者にまだ質問していない企業プロフィールキーを検出する関数
        - conversation_log: 候補者との会話ログ（リスト, 質問と回答の両方を含む）
        - company_profile: 企業情報（dict, keyが評価対象）
        - return: 未質問のキーリスト
        """

        # 面接官が質問した内容をまとめる
        asked_questions_text = "\n".join(
            [f"- {entry['question']}" for entry in conversation_log if "question" in entry]
        )

        # 企業情報キー一覧
        company_keys = list(company_profile.keys())

        # プロンプト作成
        prompt = f"""
    あなたは面接官です。
    以下の面接官の質問ログと企業プロフィールを参照し、
    まだ質問に使われていない企業プロフィールのキーを特定してください。

    企業プロフィールのキー一覧:
    {company_keys}

    面接官の質問ログ:
    {asked_questions_text}

    出力フォーマットは必ずJSONのみ:
    {{
      "unasked_keys": ["キー1", "キー2", ...]
    }}
        """

        # LLM呼び出し
        response = self._generate_response(prompt, max_tokens=512)

        # JSONとしてパース
        try:
            result = json.loads(response)
            unasked_keys = result.get("unasked_keys", [])
        except Exception:
            unasked_keys = []

        return unasked_keys
    
    def _detect_missing_keys(self, conversation_log, company_profile):
        """
        LLMを使って候補者がまだ言及していない企業プロフィールのキーを検出する
        - conversation_log: 会話ログ（候補者の回答を含む）
        - company_profile: dict（企業情報, keyが評価対象）
        - return: 欠損キーのリスト
        """

        # 候補者の回答をまとめる
        answers_text = "\n".join(
            [f"- {entry.get('answer','')}" for entry in conversation_log if "answer" in entry]
        )

        company_keys = list(company_profile.keys())

        # プロンプト
        prompt = f"""
あなたは面接官です。
以下の候補者の回答ログと企業プロフィールを参照し、
候補者が本来触れるべきだったのに言及していないキーを特定してください。

企業プロフィールのキー一覧:
{company_keys}

候補者の回答ログ:
{answers_text}

出力フォーマットは必ずJSONのみ:
{{
  "missing_keys": ["キー1", "キー2", ...]
}}
        """

        # LLM呼び出し
        response = self._generate_response(
            prompt, max_tokens=1024
        )

        # JSONをパースしてリストだけ返す
        try:
            result = json.loads(response)
            return result.get("missing_keys", [])
        except Exception:
            return []

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
        """評価タスク1: 最も意欲の低い候補者を選定する（情報欠損分析統合）"""
        print("--- 最終評価(1/3): 最も意欲の低い候補者の選定を開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        
        prompt = f"""あなたは最終決定権を持つ面接官です。全候補者の記録と情報欠損分析を確認し、「最も意欲が低い、あるいは準備不足と思われる候補者」を1名だけ選び、名前とその理由を簡潔に述べてください。

        # 全候補者の面接記録
        {conversation_summary}

        【評価基準】
        以下の観点から総合的に判断してください：
        1. **志望動機の明確さ**: 企業への志望理由が具体的で説得力があるか
        2. **企業研究の深さ**: 企業情報の理解度と準備の充実度
        3. **回答の詳細度**: 質問に対する回答の具体性と深さ
        4. **情報欠損度**: 重要な企業情報の理解不足や知識の欠如
        5. **一貫性**: 回答内容の一貫性と論理性

        【情報欠損度の考慮】
        - 情報欠損度が高い候補者は、企業研究が不十分で志望度が低い可能性が高い
        - ただし、情報欠損度だけでなく、回答の質や志望動機も総合的に評価する
        - 情報は豊富でも志望動機が不明確な候補者も考慮する

        出力形式:
        最も意欲の低い候補者: [氏名]
        理由: [具体的な理由を簡潔に]"""
        
        response = self._generate_response(prompt, max_tokens=8192)
        print("--- 最終評価(1/3): 選定を完了 ---")
        return response

    def rank_candidates_by_motivation(self, all_states):
        """評価タスク2: 候補者を意欲の低い順にランキング付けする（情報欠損分析統合）"""
        print("--- 最終評価(2/3): 意欲低い順ランキングを開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        
        prompt = f"""あなたは最終決定権を持つ面接官です。全候補者の記録と情報欠損分析を確認し、企業への志願度が低い、あるいは準備不足だと思われる順にランキング付けし、各順位の理由を簡潔に述べてください。

        # 全候補者の面接記録
        {conversation_summary}

        【志願度が低いランキング基準】
        以下の観点から総合的に判断してください：
        1. **志望動機の弱さ**: 企業への志望理由の具体性と説得力の無さ
        2. **企業研究の非充実度**: 企業情報の理解度が低く、準備が浅いか
        3. **回答の質の低さ**: 質問に対する回答の具体性、詳細度、一貫性の無さ
        4. **情報欠損度**: 重要な企業情報の理解不足（欠損度が高いほど低評価で）

        【情報欠損度の考慮】
        - 情報欠損度が高い候補者は、企業研究が充実しておらず、志望度が低い可能性が高い
        - 情報欠損度と志望動機の両方を総合的に評価する

        出力形式:
        1位: [氏名] (理由: ...)
        2位: [氏名] (理由: ...)
        3位: [氏名] (理由: ...)"""
        
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
            actual_possessed_keys = {key for key, value in possessed_knowledge.items() if value}
            all_company_keys = set(list(self.company.keys()))
            detect_possessed_keys = all_company_keys.difference(detected_missing_keys)

            true_positives = actual_missing_keys.intersection(detected_missing_keys)
            true_negatives = actual_possessed_keys.intersection(detect_possessed_keys)
            false_positives = detected_missing_keys.difference(actual_missing_keys)
            false_negatives = actual_missing_keys.difference(detected_missing_keys)
            tp_count, tn_count, fp_count, fn_count = len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp_count + tn_count) / len(all_company_keys) if all_company_keys else 0.0

            result = {
                "metrics": {
                    "predicted_missing_key_num": len(detected_missing_keys),
                    "precision": round(precision, 3), "recall": round(recall, 3), "f1_score": round(f1_score, 3),
                    "true_positives": tp_count, "false_positives": fp_count, "false_negatives": fn_count,
                },
                "details": {
                    "correctly_detected_gaps (TP)": list(true_positives),
                    "correctly_detected_knowns (TN)": list(true_negatives),
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
                     "metrics": {"missing_key_num": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "true_positives": 0, "false_positives": 0, "false_negatives": len(actual_missing_keys)},
                    "details": {"correctly_detected_gaps (TP)": [], "incorrectly_detected_gaps (FP)": [], "missed_gaps (FN)": list(actual_missing_keys)},
                    "note": "LLM output for this candidate was not found or failed to parse."}
        return evaluation_results

    def detect_knowledge_gaps(self, all_states, least_motivated_eval, ranking_eval):
        """評価タスク3: 知識欠損の定性分析と定量評価を同時に行う（情報欠損分析統合）"""
        print("--- 最終評価(3/3): 知識欠損の分析と精度評価を開始 ---")
        
        conversation_summary = self._format_all_conversations(all_states)
        full_company_info_str = json.dumps(self.company, ensure_ascii=False, indent=2)
        
        prompt = f"""あなたは、極めて洞察力の鋭い採用アナリストです。
        以下の「正解の企業情報」、「各候補者の面接記録」、「情報欠損分析結果」を比較し、候補者の知識の穴を特定してください。

        # 重要な注意点
        単に候補者が言及しなかったという理由だけで、知識が欠損していると結論づけないでください。質問の流れの中で、その情報に触れるのが自然な機会があったにもかかわらず、言及しなかったり、誤った情報を述べたり、曖昧に答えたりした場合にのみ「知識欠損」と判断してください。

        # 正解の企業情報 (キーと値のペア)
        ```json
        {full_company_info_str}
        ```

        # 各候補者の面接記録
        {conversation_summary}

        # あなたが以前予測した最も志願度が低い候補者について
        {least_motivated_eval}

        # あなたが以前予測した志願度が低い候補者順のランキングについて
        {ranking_eval}
        
        【情報欠損分析の活用】
        - 情報欠損度が高い候補者は、企業研究が不十分で知識の欠如が予想される
        - 情報欠損度の分析結果を参考に、より精密な知識欠損の特定を行う
        - ただし、情報欠損度だけでなく、実際の回答内容も詳細に分析する
        
        指示:
        各候補者について、以下の思考プロセスに基づき分析し、指定の形式で出力してください。
        1. **思考**: 候補者の各回答を検証します。「この質問に対して、この企業情報（例：'recent_news'）に触れるのが自然だったか？」「回答が具体的か、それとも一般論に終始しているか？」「誤った情報はないか？」といった観点で、知識が欠けていると判断できる「根拠」を探します。
        2. **分析**: 上記の思考に基づき、知識が不足していると判断した理由を簡潔に記述します。情報欠損分析結果も考慮してください。
        3. **キーの列挙**: 知識不足の根拠があると判断した情報の「キー」のみをJSONのリスト形式で列挙してください。根拠がなければ、空のリスト `[]` を返してください。
        4. なお、候補者の中には知識不足がない場合もあることを考慮してください。

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