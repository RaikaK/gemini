# main.py - CLIエントリ
import argparse
import sys

from config import NUM_SIMULATIONS, MAX_ROUNDS, INTERVIEWER_MODEL_TYPE, AVAILABLE_LOCAL_MODELS
from runner import run_interviews


def parse_args():
    parser = argparse.ArgumentParser(description='面接ロールプレイ実行システム')
    parser.add_argument('-n', '--num-simulations', type=int, default=NUM_SIMULATIONS,
                        help=f'シミュレーション実行回数 (デフォルト: {NUM_SIMULATIONS})')
    parser.add_argument('-s', '--set-index', type=int, default=None,
                        help='使用するデータセットのインデックス（指定しない場合はランダム）')
    parser.add_argument('--set-start', type=int, default=None,
                        help='セット番号の開始インデックス（未指定時は0）。--set-indexを指定した場合は無視される。')
    parser.add_argument('--set-end', type=int, default=None,
                        help='セット番号の終了インデックス（未指定時はデータ件数-1）。--set-indexを指定した場合は無視される。')
    parser.add_argument('-t', '--model-type', type=str, choices=['api', 'local'], default=INTERVIEWER_MODEL_TYPE,
                        help=f'面接官モデルタイプ: api または local (デフォルト: {INTERVIEWER_MODEL_TYPE})')
    parser.add_argument('-m', '--model-name', type=str, default=None,
                        help='面接官モデル名（apiの場合はOpenAI/Googleモデル、localの場合はAVAILABLE_LOCAL_MODELSのキー）')
    parser.add_argument('--list-models', action='store_true', help='利用可能なローカルモデル一覧を表示して終了')
    parser.add_argument('-r', '--max-rounds', type=int, default=None,
                        help=f'面接ラウンド数 (デフォルト: {MAX_ROUNDS})')
    parser.add_argument('--api-provider', type=str, choices=['openai', 'google'], default=None,
                        help='使用するAPIプロバイダー (openai または google)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        # 遅延インポートで依存を限定
        try:
            from model_manager import HuggingFaceModelManager
            model_manager = HuggingFaceModelManager()
            model_manager.print_model_status()
        except Exception:
            print("ローカルモデル管理モジュールが利用できません")
        return

    if args.num_simulations < 1:
        print("エラー: シミュレーション実行回数は1以上である必要があります")
        sys.exit(1)

    if args.model_type == 'local' and args.model_name and args.model_name not in AVAILABLE_LOCAL_MODELS:
        print(f"エラー: 未知のローカルモデル '{args.model_name}'")
        print(f"利用可能なモデル: {', '.join(AVAILABLE_LOCAL_MODELS.keys())}")
        print("利用可能なモデル一覧を表示するには: python main.py --list-models")
        sys.exit(1)

    run_interviews(
        num_simulations=args.num_simulations,
        set_index=args.set_index,
        set_start=args.set_start,
        set_end=args.set_end,
        interviewer_model_type=args.model_type,
        interviewer_model_name=args.model_name,
        max_rounds=args.max_rounds,
        api_provider=args.api_provider,
    )


if __name__ == '__main__':
    main()
