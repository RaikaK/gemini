# metrics.py - 評価関連のユーティリティ

import re


def calculate_ranking_accuracy(candidate_states, ranking_eval):
    """
    ランキング評価の精度指標を計算（改良版）
    正しくランキングが抽出できた場合のみスコアを計算する
    """
    try:
        true_ranking = []
        candidate_names = []
        for i, state in enumerate(candidate_states):
            profile = state['profile']
            preparation = profile.get('preparation', 'low')
            preparation_levels = {'low': 1, 'medium': 2, 'high': 3}
            motivation_score = preparation_levels.get(preparation, 1)
            candidate_name = profile.get('name', f'Candidate_{i+1}')
            true_ranking.append({
                'name': candidate_name,
                'score': motivation_score,
                'preparation': preparation
            })
            candidate_names.append(candidate_name)

        true_ranking.sort(key=lambda x: x['score'])
        true_names = [item['name'] for item in true_ranking]

        predicted_ranking = []
        if isinstance(ranking_eval, str):
            extracted_names = {}

            markdown_pattern1 = r'\*{2,}\s*(\d+)\s*位\s*\*{0,}\s*[:：]\s*([^\n\*]+?)(?:\n|$|\*|理由)'
            for match in re.finditer(markdown_pattern1, ranking_eval):
                rank_num = int(match.group(1))
                name = match.group(2).strip()
                bracket_match = re.search(r'\(([^)]+)\)', name)
                if bracket_match:
                    name = bracket_match.group(1).strip()
                name = re.sub(r'^候補者\d+\s*', '', name).strip()
                if name and 1 <= rank_num <= len(candidate_states):
                    extracted_names[rank_num] = name

            if not extracted_names:
                comma_pattern = r'((?:学生|學生)[A-Z]{1,3}\d{0,2})[、,，\s]*'
                matches = re.findall(comma_pattern, ranking_eval)
                if len(matches) >= len(candidate_states):
                    normalized_matches = [m.replace('學生', '学生') for m in matches[:len(candidate_states)]]
                    for i, name in enumerate(normalized_matches, 1):
                        extracted_names[i] = name.strip()

            if not extracted_names:
                space_pattern = r'(\d+)\.\s*(学生|學生)\s*([A-Z]{1,3})\s*(\d{0,2})'
                for match in re.finditer(space_pattern, ranking_eval):
                    rank_num = int(match.group(1))
                    name = f"学生{match.group(3)}{match.group(4)}".strip()
                    if name and 1 <= rank_num <= len(candidate_states):
                        extracted_names[rank_num] = name

            markdown_pattern2 = r'\*{2,}\s*(\d+)\.\s*([^\n\(\)\*]+?)(?:\s*\([^)]*\))?\s*\*{0,}'
            for match in re.finditer(markdown_pattern2, ranking_eval):
                rank_num = int(match.group(1))
                name = re.sub(r'^候補者\d+\s*', '', match.group(2).strip())
                if name and 1 <= rank_num <= len(candidate_states):
                    extracted_names[rank_num] = name

            for i in range(1, len(candidate_states) + 1):
                predicted_ranking.append(extracted_names.get(i, "不明"))
        else:
            predicted_ranking = ["不明"] * len(candidate_states)

        def normalize_name(name):
            if not name or name == "不明":
                return None
            return re.sub(r'[\s()（）、,，。*]', '', name) or None

        normalized_candidate_names = {normalize_name(name): name for name in candidate_names if normalize_name(name)}

        matched_ranking = []
        for pred_name in predicted_ranking:
            normalized_pred = normalize_name(pred_name)
            if normalized_pred and normalized_pred in normalized_candidate_names:
                matched_ranking.append(normalized_candidate_names[normalized_pred])
            else:
                matched = False
                for norm_cand_name, orig_cand_name in normalized_candidate_names.items():
                    if normalized_pred and (normalized_pred in norm_cand_name or norm_cand_name in normalized_pred):
                        matched_ranking.append(orig_cand_name)
                        matched = True
                        break
                if not matched:
                    matched_ranking.append("不明")

        is_valid = "不明" not in matched_ranking and len(set(matched_ranking)) == len(matched_ranking)
        if not is_valid:
            return {
                'accuracy': None,
                'is_valid': False,
                'true_ranking': true_ranking,
                'predicted_ranking': matched_ranking,
                'raw_predicted_ranking': predicted_ranking,
                'message': 'ランキングが正しく抽出できませんでした。スコアは計算されません。'
            }

        total_pairs = 0
        correct_pairs = 0
        n = len(true_names)
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                try:
                    pred_idx_i = matched_ranking.index(true_names[i])
                    pred_idx_j = matched_ranking.index(true_names[j])
                except ValueError:
                    continue
                if pred_idx_i < pred_idx_j:
                    correct_pairs += 1

        ranking_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
        correct_positions = sum(1 for true, pred in zip(true_names, matched_ranking) if true == pred)

        return {
            'accuracy': ranking_accuracy,
            'is_valid': True,
            'true_ranking': true_ranking,
            'predicted_ranking': matched_ranking,
            'raw_predicted_ranking': predicted_ranking,
            'correct_pairs': correct_pairs,
            'total_pairs': total_pairs,
            'correct_positions': correct_positions,
            'total_positions': len(true_names)
        }

    except Exception as e:
        return {
            'accuracy': None,
            'is_valid': False,
            'error': str(e),
            'message': f'ランキング精度の計算中にエラーが発生しました: {e}'
        }


def calculate_knowledge_gaps_metrics(candidate_states, knowledge_gaps_eval):
    """評価3: 知識欠損検出の精度指標を計算"""
    try:
        if not knowledge_gaps_eval or 'quantitative_performance_metrics' not in knowledge_gaps_eval:
            return None

        performance_metrics = knowledge_gaps_eval['quantitative_performance_metrics']
        accuracies, f1_scores, precisions, recalls = [], [], [], []

        for state in candidate_states:
            candidate_name = state['profile'].get('name')
            if candidate_name in performance_metrics:
                metrics = performance_metrics[candidate_name].get('metrics', {})
                if 'accuracy' in metrics:
                    accuracies.append(metrics['accuracy'])
                if 'f1_score' in metrics:
                    f1_scores.append(metrics['f1_score'])
                if 'precision' in metrics:
                    precisions.append(metrics['precision'])
                if 'recall' in metrics:
                    recalls.append(metrics['recall'])

        kg_accuracy_by_motivation = {'low': None, 'medium': None, 'high': None}
        for state in candidate_states:
            candidate_name = state['profile'].get('name')
            preparation_level = state['profile'].get('preparation', 'low')
            if candidate_name in performance_metrics:
                metrics = performance_metrics[candidate_name].get('metrics', {})
                if preparation_level in kg_accuracy_by_motivation and 'accuracy' in metrics:
                    kg_accuracy_by_motivation[preparation_level] = metrics['accuracy']

        return {
            'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0,
            'avg_f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            'avg_precision': sum(precisions) / len(precisions) if precisions else 0.0,
            'avg_recall': sum(recalls) / len(recalls) if recalls else 0.0,
            'knowledge_gaps_metrics_by_motivation': kg_accuracy_by_motivation,
            'per_candidate_metrics': performance_metrics
        }

    except Exception:
        return None


def get_last_common_question_evaluations(round_evaluations):
    """最後の全体質問ラウンドの評価結果を取得"""
    for eval_data in reversed(round_evaluations):
        if eval_data.get('question_type') == 'common' and eval_data.get('evaluations') is not None:
            return eval_data['evaluations']
    return {'least_motivated': None, 'ranking': None, 'knowledge_gaps': None}


def get_last_common_question_ranking_accuracy(round_evaluations):
    """最後の全体質問ラウンドのランキング精度を取得"""
    for eval_data in reversed(round_evaluations):
        if eval_data.get('question_type') == 'common':
            return eval_data.get('ranking_accuracy')
    return None


def get_last_common_question_knowledge_gaps_metrics(round_evaluations):
    """最後の全体質問ラウンドの知識欠損メトリクスを取得"""
    for eval_data in reversed(round_evaluations):
        if eval_data.get('question_type') == 'common':
            return eval_data.get('knowledge_gaps_metrics')
    return None
