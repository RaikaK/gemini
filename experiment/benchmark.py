# benchmark.py

import os
import json
import glob
from utils import call_gemini_api, parse_json_from_response
from config import GENERATOR_MODEL_NAME

class BenchmarkEvaluator:
    """
    学生の回答と面接官の質問を分析し、ベンチマーク評価を行うクラス。
    """
    def __init__(self, full_company_profile):
        self.full_company_profile = full_company_profile
        self.evaluator_model_name = GENERATOR_MODEL_NAME

    def _create_student_evaluation_prompt(self, candidate_knowledge, question, answer):
        """学生の回答補完能力を評価するためのプロンプトを生成する"""
        prompt = f"""
        あなたは、学生の回答を分析する客観的な評価官です。
        以下の情報に基づき、学生の回答が情報の欠損をどれだけうまく補完できたかを評価してください。

        # 1. 正解となる完全な企業情報
        {json.dumps(self.full_company_profile, ensure_ascii=False, indent=2)}

        # 2. 学生が事前に与えられていた、一部欠損した企業情報
        {json.dumps(candidate_knowledge, ensure_ascii=False, indent=2)}

        # 3. 面接官からの質問
        "{question}"

        # 4. 学生の回答
        "{answer}"

        # 評価タスク
        上記の4つの情報を比較し、以下の基準で学生の回答を評価してください。
        特に、学生が知らなかったはずの情報（情報2にはなく、情報1には存在する情報）について、どれだけもっともらしく、矛盾なく回答できたかに注目してください。

        # 出力形式 (JSON)
        ```json
        {{
          "factual_accuracy": {{
            "score": [1-5の整数],
            "reason": "事実との整合性に関する評価理由（例：既知の情報と矛盾はないか、誤った情報を断定していないか）"
          }},
          "plausible_completion": {{
            "score": [1-5の整数],
            "reason": "知らない情報の補完のもっともらしさに関する評価理由（例：業界の一般論として自然か、文脈に合っているか）"
          }},
          "hallucination_level": {{
            "level": ["low", "medium", "high"],
            "reason": "ハルシネーション（幻覚）の度合いに関する評価理由（例：事実でない情報を断定的に述べているか）"
          }}
        }}
        ```
        """
        return prompt
    
    def _create_interviewer_evaluation_prompt(self, candidate_knowledge, question):
        """面接官の質問品質を評価するためのプロンプトを生成する"""
        prompt = f"""
        あなたは、面接官の質問の質を分析する客観的な評価官です。
        以下の情報に基づき、面接官の質問が、学生が知らないであろう情報を効果的に突いているかを評価してください。

        # 1. 正解となる完全な企業情報
        {json.dumps(self.full_company_profile, ensure_ascii=False, indent=2)}

        # 2. 学生が事前に与えられていた、一部欠損した企業情報
        # (空欄 "" になっている項目が、学生が知らない情報です)
        {json.dumps(candidate_knowledge, ensure_ascii=False, indent=2)}

        # 3. 面接官がした質問
        "{question}"

        # 評価タスク
        上記の情報を比較し、面接官の質問が、学生が知らない情報（情報2で値が空 "" になっている項目）に意図的に関連している度合いを評価してください。

        # 出力形式 (JSON)
        ```json
        {{
          "targeted_probing_score": {{
            "score": [1-5の整数。5が高いほど鋭い質問],
            "reason": "評価理由（例：学生が知らない「awards」について巧みに質問しているため高評価。既知の情報に関する質問のため低評価。）"
          }},
          "targeted_topic": "質問が狙ったと思われる、学生が知らない情報の項目名（例：「awards」）。該当しない場合は null。"
        }}
        ```
        """
        return prompt

    def evaluate_student_completion(self, candidate_knowledge, conversation_log):
        """会話ログを分析し、学生の回答補完能力を評価する"""
        turn_evaluations = []
        print("--- 学生の補完能力ベンチマーク評価を開始 ---")
        for i, turn in enumerate(conversation_log):
            print(f"  ターン {i+1}/{len(conversation_log)} の回答を評価中...")
            prompt = self._create_student_evaluation_prompt(
                candidate_knowledge, turn["question"], turn["answer"]
            )
            response = call_gemini_api(self.evaluator_model_name, prompt)
            evaluation = parse_json_from_response(response)
            turn_evaluations.append({
                "turn": turn["turn"],
                "evaluation": evaluation
            })
        print("--- 学生の補完能力ベンチマーク評価完了 ---")
        return {"turn_by_turn_evaluation": turn_evaluations}

    def evaluate_interviewer_performance(self, candidate_knowledge, conversation_log, interview_flow):
        """会話ログを分析し、面接官が学生の知識の穴を突く質問をできたかを評価する"""
        interviewer_evaluations = []
        print("--- 面接官の質問品質ベンチマーク評価を開始 ---")
        
        for i, turn in enumerate(conversation_log):
            # 個別質問(1)のラウンドのみを評価対象とする
            if i < len(interview_flow) and interview_flow[i] == 1:
                print(f"  個別質問ターン {i+1}/{len(conversation_log)} の質問を評価中...")
                prompt = self._create_interviewer_evaluation_prompt(
                    candidate_knowledge, turn["question"]
                )
                response = call_gemini_api(self.evaluator_model_name, prompt)
                evaluation = parse_json_from_response(response)
                interviewer_evaluations.append({
                    "turn": turn["turn"],
                    "question": turn["question"],
                    "evaluation": evaluation
                })
        
        print("--- 面接官の質問品質ベンチマーク評価完了 ---")
        return {"individual_question_evaluation": interviewer_evaluations}

# ==============================================================================
# 複数実験の集計・分析機能
# ==============================================================================

def analyze_all_results(results_dir="results"):
    """
    指定されたディレクトリ内の全ての実験結果JSONファイルを読み込み、
    学生の準備レベルごとの平均評価点を集計・分析する。
    """
    json_files = glob.glob(os.path.join(results_dir, "experiment_*.json"))

    if not json_files:
        print(f"エラー: '{results_dir}' ディレクトリに実験結果ファイルが見つかりません。")
        return

    # 準備レベルごとにスコアを格納するための辞書
    scores_by_prep_level = {
        "high": [],
        "medium": [],
        "low": []
    }

    # 各JSONファイルを処理
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            interview_results = data.get("interview_results", [])
            for result in interview_results:
                candidate_info = result.get("candidate_info", {})
                prep_level = candidate_info.get("preparation")
                
                # 'high-middle'は'medium'として集計する
                if prep_level == "high-middle":
                    prep_level = "medium"

                if prep_level in scores_by_prep_level:
                    # 面接官の評価を取得
                    interviewer_eval = result.get("interviewer_evaluation", {})
                    overall_score = interviewer_eval.get("overall_score")
                    
                    if isinstance(overall_score, (int, float)):
                        scores_by_prep_level[prep_level].append(overall_score)

    # 平均点の計算と結果の表示
    print(f"\n--- 実験結果の集計 ({len(json_files)}件のファイル) ---")
    
    for level, scores in scores_by_prep_level.items():
        if scores:
            average_score = sum(scores) / len(scores)
            print(f"準備レベル '{level}' の学生の平均評価点: {average_score:.2f} / 5.0  ({len(scores)}回の面接)")
        else:
            print(f"準備レベル '{level}' の学生のデータはありませんでした。")

    print("\n--- 分析 ---")
    if scores_by_prep_level["high"] and scores_by_prep_level["low"]:
        avg_high = sum(scores_by_prep_level["high"]) / len(scores_by_prep_level["high"])
        avg_low = sum(scores_by_prep_level["low"]) / len(scores_by_prep_level["low"])
        
        if avg_high > avg_low + 1.0:
            print("面接官は、学生の知識レベルの差を比較的よく見抜けているようです。")
        elif avg_high > avg_low:
            print("面接官は知識レベルの差をある程度見抜けていますが、差は大きくありません。")
        else:
            print("面接官は学生の知識レベルの差をほとんど見抜けず、知識の少ない学生に騙されている可能性が高いです。")

if __name__ == "__main__":
    # このファイルが直接実行された場合に、集計・分析機能を開始する
    analyze_all_results()
