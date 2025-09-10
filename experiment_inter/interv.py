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

    def should_continue_interview(self, conversation_history, current_round, max_rounds=5):
        """面接を続けるべきかどうかを判断する"""
        if current_round >= max_rounds:
            return False, "最大ラウンド数に達しました"
        
        if not conversation_history:
            return True, "初回のため継続"
        
        # 最近の回答を分析して自信度を判定
        recent_answers = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent_answers])
        
        # 企業情報の項目リストを取得
        company_keys = list(self.company.keys())
        company_keys_str = ", ".join(company_keys)
        
        prompt = f"""あなたは、{self.company.get('name')}の採用面接官です。
        現在の面接ラウンド: {current_round}/{max_rounds}
        
        以下の最近の会話履歴を分析し、候補者の志望度について十分な情報が得られているか判断してください。
        
        会話履歴:
        {history_str}
        
        企業情報の項目リスト:
        {company_keys_str}
        
        【具体的な終了条件チェックリスト】
        以下の全ての項目について確認し、足りない項目がないことを確認してください：
        
        1. 志望度判定に必要な情報の充足度
           - 候補者の志望動機が明確に把握できているか
           - 企業への理解度（準備レベル）が判断できるか
           - 他の候補者との比較が可能な材料が揃っているか
        
        2. 企業知識の網羅性
           - 重要な企業情報項目について言及されているか
           - 候補者の知識の欠損箇所が特定できているか
           - 誤った情報や曖昧な回答がないか
        
        3. 追加質問の必要性
           - 残りの質問で新たな洞察が得られる可能性があるか
           - 現在の情報で志望度の判定が可能か
           - 他の候補者との差別化に必要な情報が不足していないか
        
        【終了判定基準】
        上記のチェックリストで「足りない項目がない」と判断できる場合のみ終了してください。
        一つでも不十分な項目がある場合は継続してください。
        
        回答形式:
        - 継続する場合: "CONTINUE"
        - 終了する場合: "STOP"
        - 理由: [具体的にどの項目が不足しているか、または十分であるかを明記]
        
        回答:"""
        
        response = self._generate_response(prompt, max_tokens=300)
        
        # レスポンスを解析
        if "CONTINUE" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "追加情報が必要"
            return True, reason
        elif "STOP" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "十分な情報が得られた"
            return False, reason
        else:
            # デフォルトは継続（安全側）
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

    def conduct_dynamic_interview(self, candidate_states, applicant, max_rounds=10):
        """智的な動的面接フローを実行する（質問回数上限を撤廃）"""
        print(f"--- 智的動的面接フロー開始 (最大{max_rounds}ラウンド) ---")
        
        asked_common_questions = []
        current_round = 0
        actual_interview_flow = []  # 実際に実行された面接フローを記録
        
        # 各候補者の個別質問回数を追跡
        individual_question_counts = {i: 0 for i in range(len(candidate_states))}
        
        while current_round < max_rounds:
            current_round += 1
            print(f"--- 面接ラウンド {current_round}/{max_rounds} ---")
            
            # 次の質問タイプを智的に決定（質問回数制限を考慮しない）
            question_type, reason = self.decide_next_question_type_enhanced(
                candidate_states, asked_common_questions, current_round, max_rounds, individual_question_counts
            )
            
            print(f"--- 選択された質問タイプ: {question_type} ---")
            print(f"--- 選択理由: {reason} ---")
            
            if question_type == "common":
                # 全体質問フェーズ
                print("--- 全体質問フェーズを実行 ---")
                actual_interview_flow.append(0)  # 0 = 全体質問
                question, _ = self.ask_common_question(asked_common_questions)
                asked_common_questions.append(question)
                print(f"--- 生成された全体質問: 「{question}」 ---")
                
                for i, state in enumerate(candidate_states):
                    print(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} へ質問")
                    # 学生の回答を生成
                    answer, token_info = applicant.generate(
                        state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                    )
                    print(f"学生 (API): {answer}")
                    print(f"Token数: {token_info['total_tokens']} (プロンプト: {token_info['prompt_tokens']}, 回答: {token_info['completion_tokens']})")
                    state["conversation_log"].append({
                        "turn": current_round, 
                        "question": question, 
                        "answer": answer,
                        "token_info": token_info
                    })
                
                # 全体質問後の継続判断
                overall_responses = [state['conversation_log'][-1] for state in candidate_states if state['conversation_log']]
                should_continue, continue_reason = self.should_continue_interview(
                    overall_responses, current_round, max_rounds
                )
                print(f"全体質問後の継続判断: {'継続' if should_continue else '終了'} - {continue_reason}")
                if not should_continue:
                    break
                    
            elif question_type == "individual":
                # 個別質問フェーズ（情報欠損検出機能統合）
                print("--- 個別質問フェーズを実行 ---")
                actual_interview_flow.append(1)  # 1 = 個別質問
                any_continued = False
                
                # 情報欠損候補者に集中すべきか判断
                should_focus, focus_reason, focus_indices = self._should_focus_on_deficient_candidates(candidate_states, individual_question_counts)
                print(f"--- 情報欠損分析: {focus_reason} ---")
                
                # 質問対象の候補者を決定
                if should_focus and focus_indices:
                    # 情報欠損候補者に集中
                    question_targets = focus_indices
                    print(f"--- 情報欠損候補者に集中: {[f'候補者{i+1}' for i in focus_indices]} ---")
                else:
                    # 全候補者に質問
                    question_targets = list(range(len(candidate_states)))
                    print("--- 全候補者に個別質問 ---")
                
                for i in question_targets:
                    state = candidate_states[i]
                    print(f"-> 候補者 {i+1}: {state['profile'].get('name', 'N/A')} への個別面接判断")
                    
                    # 個別質問の継続判断（質問回数制限を考慮しない）
                    should_continue, reason = self.should_continue_individual_interview(
                        state['conversation_log'], current_round, max_rounds, individual_question_counts[i]
                    )
                    
                    if should_continue:
                        any_continued = True
                        individual_question_counts[i] += 1
                        question, _ = self.ask_question(state['conversation_log'])
                        print(f"面接官 ({self.model_type}): {question}")
                        print(f"候補者 {i+1} の個別質問回数: {individual_question_counts[i]}回目")
                        # 学生の回答を生成
                        answer, token_info = applicant.generate(
                            state["profile"], state["knowledge_tuple"], state["conversation_log"], question
                        )
                        print(f"学生 (API): {answer}")
                        print(f"Token数: {token_info['total_tokens']} (プロンプト: {token_info['prompt_tokens']}, 回答: {token_info['completion_tokens']})")
                        state["conversation_log"].append({
                            "turn": current_round, 
                            "question": question, 
                            "answer": answer,
                            "token_info": token_info
                        })
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
        print(f"--- 各候補者の個別質問回数: {individual_question_counts} ---")
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

    def should_continue_individual_interview(self, conversation_history, current_round, max_rounds, individual_question_count):
        """個別面接の継続判断（質問回数制限なし）"""
        if current_round >= max_rounds:
            return False, "最大ラウンド数に達しました"
        
        if not conversation_history:
            return True, "初回のため継続"
        
        # 最近の回答を分析
        recent_answers = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        history_str = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent_answers])
        
        # 企業情報の項目リストを取得
        company_keys = list(self.company.keys())
        company_keys_str = ", ".join(company_keys)
        
        prompt = f"""あなたは、{self.company.get('name')}の採用面接官です。
        現在の面接ラウンド: {current_round}/{max_rounds}
        この候補者への個別質問回数: {individual_question_count}回
        
        以下の最近の会話履歴を分析し、この候補者の志望度について十分な情報が得られているか判断してください。
        
        会話履歴:
        {history_str}
        
        企業情報の項目リスト:
        {company_keys_str}
        
        【個別面接の具体的な終了条件チェックリスト】
        以下の全ての項目について確認し、足りない項目がないことを確認してください：
        
        1. 志望度判定に必要な情報の充足度
           - この候補者の志望動機が明確に把握できているか
           - 企業への理解度（準備レベル）が判断できるか
           - 他の候補者との比較が可能な材料が揃っているか
        
        2. 企業知識の網羅性
           - 重要な企業情報項目について言及されているか
           - この候補者の知識の欠損箇所が特定できているか
           - 誤った情報や曖昧な回答がないか
        
        3. 個別質問の効果性
           - 個別質問で新たな洞察が得られる可能性があるか
           - 現在の情報で志望度の判定が可能か
           - この候補者の特徴的な回答パターンが把握できているか
        
        【終了判定基準】
        上記のチェックリストで「足りない項目がない」と判断できる場合のみ終了してください。
        一つでも不十分な項目がある場合は継続してください。
        
        回答形式:
        - 継続する場合: "CONTINUE"
        - 終了する場合: "STOP"
        - 理由: [具体的にどの項目が不足しているか、または十分であるかを明記]
        
        回答:"""
        
        response = self._generate_response(prompt, max_tokens=300)
        
        # レスポンスを解析
        if "CONTINUE" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "追加情報が必要"
            return True, reason
        elif "STOP" in response.upper():
            reason = response.split("理由:")[-1].strip() if "理由:" in response else "十分な情報が得られた"
            return False, reason
        else:
            # デフォルトは継続（安全側）
            return True, "判断不明のため継続"

    def decide_next_question_type_enhanced(self, candidate_states, asked_common_questions, current_round, max_rounds, individual_question_counts):
        """次の質問タイプを智的に決定する（質問回数制限を考慮しない拡張版）"""
        
        # 基本的な条件チェック
        if current_round == 1:
            return "common", "初回は全体質問から開始"
        
        if len(asked_common_questions) == 0:
            return "common", "まだ全体質問を行っていない"
        
        # 全候補者の回答状況を分析
        all_responses_count = sum(len(state['conversation_log']) for state in candidate_states)
        avg_responses_per_candidate = all_responses_count / len(candidate_states) if candidate_states else 0
        
        # 個別質問回数の統計
        total_individual_questions = sum(individual_question_counts.values())
        avg_individual_questions = total_individual_questions / len(candidate_states) if candidate_states else 0
        
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
        
        # 情報欠損候補者の分析
        should_focus, focus_reason, focus_indices = self._should_focus_on_deficient_candidates(candidate_states, individual_question_counts)
        deficient_candidates = self._identify_deficient_candidates(candidate_states)
        
        # LLMによる状況分析
        situation_summary = self._analyze_interview_situation_enhanced(candidate_states, asked_common_questions, current_round, individual_question_counts)
        
        # 欠損候補者の詳細情報
        deficiency_info = "\n".join([f"- {c['name']}: 欠損度{c['deficiency_score']:.2f} ({c['reason']})" for c in deficient_candidates])
        
        prompt = f"""あなたは、{self.company.get('name')}の面接戦略エキスパートです。
        現在の面接状況を分析し、次に行うべき質問タイプを決定してください。

        # 現在の状況
        - 現在のラウンド: {current_round}/{max_rounds}
        - 実施済み全体質問数: {len(asked_common_questions)}
        - 候補者あたり平均回答数: {avg_responses_per_candidate:.1f}
        - 候補者あたり平均個別質問数: {avg_individual_questions:.1f}
        - 最後の全体質問からの経過ラウンド: {rounds_since_common}
        
        # 情報欠損分析
        欠損候補者の分析: {should_focus}
        集中対象: {focus_reason}
        欠損候補者詳細:
        {deficiency_info}
        
        # 状況分析
        {situation_summary}
        
        # 判断基準（情報欠損検出機能統合）
        【全体質問を選ぶべき場合】
        - 候補者間の比較材料が不足している
        - 特定の重要なトピックについて全員の見解が必要
        - 個別質問で深掘りする前に基盤となる情報が必要
        - 最後の全体質問から時間が経ちすぎている（3ラウンド以上）
        - 全候補者の知識レベルを均等に把握する必要がある
        - 情報欠損候補者への個別質問が効果的でない場合
        
        【個別質問を選ぶべき場合】
        - 特定の候補者の回答をより深く探る必要がある
        - 候補者ごとに異なる角度からの質問が効果的
        - 志望度の判定に必要な個人的な動機を探る必要がある
        - 十分な全体質問が既に実施されている
        - 情報が不足している候補者が特定されている（欠損度>0.6）
        - 個別質問の効果が期待できる状況
        - 情報欠損候補者に集中すべき場合
        
        【情報欠損に基づく判断】
        - 欠損度が高い候補者がいる場合は個別質問を優先
        - 個別質問で改善が見られない場合は全体質問に切り替え
        - 複数の候補者が欠損している場合は効率的な全体質問を検討
        
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

    def _analyze_interview_situation_enhanced(self, candidate_states, asked_common_questions, current_round, individual_question_counts):
        """現在の面接状況を分析して要約を作成（質問回数制限を考慮しない拡張版）"""
        
        # 各候補者の回答の特徴を分析
        candidate_analysis = []
        for i, state in enumerate(candidate_states):
            name = state['profile'].get('name', f'候補者{i+1}')
            response_count = len(state['conversation_log'])
            individual_count = individual_question_counts.get(i, 0)
            
            # 最新の回答の長さと内容の傾向を分析
            if state['conversation_log']:
                recent_answers = [turn['answer'] for turn in state['conversation_log'][-2:]]
                avg_answer_length = sum(len(answer) for answer in recent_answers) / len(recent_answers)
                
                # 回答の詳細度を簡易評価
                detail_level = "高い" if avg_answer_length > 100 else "中程度" if avg_answer_length > 50 else "低い"
                
                # 情報の充実度を評価
                info_richness = "豊富" if response_count >= 4 and avg_answer_length > 80 else "標準" if response_count >= 2 else "限定的"
                
                candidate_analysis.append(f"- {name}: 回答数{response_count}, 個別質問{individual_count}回, 詳細度{detail_level}, 情報量{info_richness}")
            else:
                candidate_analysis.append(f"- {name}: 未回答")
        
        # 質問のバランス分析
        common_questions_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(asked_common_questions))
        
        # 個別質問の分布分析
        individual_distribution = f"個別質問回数分布: {dict(individual_question_counts)}"
        
        situation = f"""
候補者別の回答状況:
{chr(10).join(candidate_analysis)}

実施済み全体質問:
{common_questions_str if common_questions_str else "  なし"}

{individual_distribution}

現在の課題:
- 候補者間の比較材料の充実度
- 各候補者の志望度判定に必要な情報の蓄積状況
- 個別質問の効果性と情報収集の効率性
- 面接の残り時間と効率性のバランス
- 情報が不足している候補者の特定
        """
        
        return situation.strip()

    def _evaluate_candidate_information_deficiency(self, candidate_state, company_keys):
        """候補者の情報欠損度を評価する"""
        conversation_log = candidate_state['conversation_log']
        if not conversation_log:
            return 1.0, "会話履歴なし"  # 最大の欠損度
        
        # 回答の長さと内容の豊富さを分析
        total_answer_length = sum(len(turn['answer']) for turn in conversation_log)
        avg_answer_length = total_answer_length / len(conversation_log)
        
        # 企業情報項目の言及度を分析
        mentioned_keys = set()
        for turn in conversation_log:
            answer = turn['answer'].lower()
            for key in company_keys:
                if key.lower() in answer:
                    mentioned_keys.add(key)
        
        mention_ratio = len(mentioned_keys) / len(company_keys) if company_keys else 0
        
        # 回答の詳細度を評価
        detail_score = min(avg_answer_length / 100, 1.0)  # 100文字を基準とした詳細度
        
        # 総合的な欠損度を計算（0-1の範囲、1が最大の欠損）
        deficiency_score = 1.0 - (mention_ratio * 0.6 + detail_score * 0.4)
        
        # 欠損度の説明
        if deficiency_score > 0.8:
            deficiency_level = "高い"
        elif deficiency_score > 0.5:
            deficiency_level = "中程度"
        else:
            deficiency_level = "低い"
        
        reason = f"言及率{mention_ratio:.2f}, 詳細度{detail_score:.2f}, 欠損度{deficiency_level}"
        
        return deficiency_score, reason

    def _identify_deficient_candidates(self, candidate_states):
        """情報欠損が疑われる候補者を特定する"""
        company_keys = list(self.company.keys())
        candidate_deficiencies = []
        
        for i, state in enumerate(candidate_states):
            deficiency_score, reason = self._evaluate_candidate_information_deficiency(state, company_keys)
            candidate_deficiencies.append({
                'index': i,
                'name': state['profile'].get('name', f'候補者{i+1}'),
                'deficiency_score': deficiency_score,
                'reason': reason
            })
        
        # 欠損度でソート（高い順）
        candidate_deficiencies.sort(key=lambda x: x['deficiency_score'], reverse=True)
        
        return candidate_deficiencies

    def _should_focus_on_deficient_candidates(self, candidate_states, individual_question_counts):
        """情報欠損が疑われる候補者に個別質問を集中すべきか判断する"""
        deficient_candidates = self._identify_deficient_candidates(candidate_states)
        
        # 最も欠損度の高い候補者を特定
        most_deficient = deficient_candidates[0] if deficient_candidates else None
        
        if not most_deficient:
            return False, "欠損候補者なし", []
        
        # 欠損度が高い候補者の個別質問回数を確認
        most_deficient_index = most_deficient['index']
        individual_count = individual_question_counts.get(most_deficient_index, 0)
        
        # 欠損度が高く、まだ個別質問が少ない場合は集中すべき
        if most_deficient['deficiency_score'] > 0.6 and individual_count < 3:
            return True, f"候補者{most_deficient_index+1}({most_deficient['name']})に集中", [most_deficient_index]
        
        # 複数の候補者が欠損している場合は、上位2-3名に集中
        high_deficiency_candidates = [c for c in deficient_candidates if c['deficiency_score'] > 0.5]
        if len(high_deficiency_candidates) >= 2:
            focus_indices = [c['index'] for c in high_deficiency_candidates[:3]]
            return True, f"複数の欠損候補者に集中", focus_indices
        
        return False, "個別質問集中の必要なし", []

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
        
        # 情報欠損分析を実行
        company_keys = list(self.company.keys())
        deficiency_analysis = []
        for i, state in enumerate(all_states):
            deficiency_score, reason = self._evaluate_candidate_information_deficiency(state, company_keys)
            name = state['profile'].get('name', f'候補者{i+1}')
            deficiency_analysis.append(f"- {name}: 情報欠損度 {deficiency_score:.2f} ({reason})")
        
        deficiency_summary = "\n".join(deficiency_analysis)
        
        prompt = f"""あなたは最終決定権を持つ面接官です。全候補者の記録と情報欠損分析を確認し、「最も意欲が低い、あるいは準備不足と思われる候補者」を1名だけ選び、名前とその理由を簡潔に述べてください。

        # 全候補者の面接記録
        {conversation_summary}

        # 情報欠損分析結果
        {deficiency_summary}

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
        """評価タスク2: 候補者を意欲順にランキング付けする（情報欠損分析統合）"""
        print("--- 最終評価(2/3): 意欲順ランキングを開始 ---")
        conversation_summary = self._format_all_conversations(all_states)
        
        # 情報欠損分析を実行
        company_keys = list(self.company.keys())
        deficiency_analysis = []
        for i, state in enumerate(all_states):
            deficiency_score, reason = self._evaluate_candidate_information_deficiency(state, company_keys)
            name = state['profile'].get('name', f'候補者{i+1}')
            deficiency_analysis.append(f"- {name}: 情報欠損度 {deficiency_score:.2f} ({reason})")
        
        deficiency_summary = "\n".join(deficiency_analysis)
        
        prompt = f"""あなたは最終決定権を持つ面接官です。全候補者の記録と情報欠損分析を確認し、企業への意欲が高い順にランキング付けし、各順位の理由を簡潔に述べてください。

        # 全候補者の面接記録
        {conversation_summary}

        # 情報欠損分析結果
        {deficiency_summary}

        【ランキング基準】
        以下の観点から総合的に判断してください：
        1. **志望動機の強さ**: 企業への志望理由の具体性と説得力
        2. **企業研究の充実度**: 企業情報の理解度と準備の深さ
        3. **回答の質**: 質問に対する回答の具体性、詳細度、一貫性
        4. **情報欠損度**: 重要な企業情報の理解不足（欠損度が低いほど高評価）
        5. **熱意の表現**: 企業への関心と意欲の表現度

        【情報欠損度の考慮】
        - 情報欠損度が低い候補者は、企業研究が充実しており志望度が高い可能性が高い
        - ただし、情報は豊富でも志望動機が不明確な候補者は順位を下げる
        - 情報欠損度と志望動機の両方を総合的に評価する

        出力形式:
        1位: [氏名] (理由: ...)
        2位: [氏名] (理由: ...)
        3位: [氏名] (理由: ...)"""
        
        response = self._generate_response(prompt, max_tokens=8192)
        print("--- 最終評価(2/3): ランキングを完了 ---")
        return response

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
        """評価タスク3: 知識欠損の定性分析と定量評価を同時に行う（情報欠損分析統合）"""
        print("--- 最終評価(3/3): 知識欠損の分析と精度評価を開始 ---")
        
        conversation_summary = self._format_all_conversations(all_states)
        full_company_info_str = json.dumps(self.company, ensure_ascii=False, indent=2)
        
        # 情報欠損分析を実行
        company_keys = list(self.company.keys())
        deficiency_analysis = []
        for i, state in enumerate(all_states):
            deficiency_score, reason = self._evaluate_candidate_information_deficiency(state, company_keys)
            name = state['profile'].get('name', f'候補者{i+1}')
            deficiency_analysis.append(f"- {name}: 情報欠損度 {deficiency_score:.2f} ({reason})")
        
        deficiency_summary = "\n".join(deficiency_analysis)
        
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

        # 情報欠損分析結果
        {deficiency_summary}
        
        【情報欠損分析の活用】
        - 情報欠損度が高い候補者は、企業研究が不十分で知識の欠如が予想される
        - 情報欠損度の分析結果を参考に、より精密な知識欠損の特定を行う
        - ただし、情報欠損度だけでなく、実際の回答内容も詳細に分析する
        
        指示:
        各候補者について、以下の思考プロセスに基づき分析し、指定の形式で出力してください。
        1. **思考**: 候補者の各回答を検証します。「この質問に対して、この企業情報（例：'recent_news'）に触れるのが自然だったか？」「回答が具体的か、それとも一般論に終始しているか？」「誤った情報はないか？」といった観点で、知識が欠けていると判断できる「根拠」を探します。
        2. **分析**: 上記の思考に基づき、知識が不足していると判断した理由を簡潔に記述します。情報欠損分析結果も考慮してください。
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