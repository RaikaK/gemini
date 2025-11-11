import json
import os
from pathlib import Path

def find_invalid_ranking_files(root_dir: str) -> list[str]:
    """
    指定されたディレクトリ以下のJSONファイルを再帰的に検索し、
    "predicted_ranking"に無効な値（"不明"または"**"）が含まれる
    ファイルのパスのリストを返します。

    Args:
        root_dir (str): 検索を開始するディレクトリのパス。

    Returns:
        list[str]: 無効なランキングを含むJSONファイルのパスのリスト。
    """
    
    invalid_files = []
    
    # 検索対象とする無効な値
    INVALID_VALUES = {"不明", "**"} 
    
    # pathlibを使ってディレクトリ以下のすべてのJSONファイルを再帰的に検索
    for file_path in Path(root_dir).rglob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 必要なキーへのアクセスを試みる
            ranking_data = data.get("accuracy_metrics", {}).get("ranking_accuracy", {})
            predicted_ranking = ranking_data.get("predicted_ranking")

            if predicted_ranking and isinstance(predicted_ranking, list):
                # predicted_rankingリスト内のいずれかの要素が無効値に含まれるかチェック
                if any(item in INVALID_VALUES for item in predicted_ranking):
                    # 絶対パスではなく、実行場所からの相対パスとして記録
                    invalid_files.append(str(file_path))
                    
        except json.JSONDecodeError:
            print(f"警告: {file_path} は無効なJSONファイルです。スキップします。")
        except Exception as e:
            print(f"警告: {file_path} の処理中に予期せぬエラーが発生しました: {e}")

    return invalid_files

# --- 実行部分 ---
if __name__ == "__main__":
    # 🔍 検索対象ディレクトリのパスを編集してください
    # (例: スクリプトと同じディレクトリを検索する場合、"." を指定)
    SEARCH_DIRECTORY = "/home/yanai-lab/karasawa-k/bootcamp/penguin-paper/experiment_inter/results" 
    
    print(f"▶️ 検索ディレクトリ: {os.path.abspath(SEARCH_DIRECTORY)}")
    print("-" * 30)
    
    result_files = find_invalid_ranking_files(SEARCH_DIRECTORY)

    if result_files:
        print(f"🚨 無効な予測ランキングを含むファイル ({len(result_files)} 件):")
        for filename in result_files:
            print(f"- {filename}")
    else:
        print("✅ 無効な予測ランキングを含むファイルは見つかりませんでした。")