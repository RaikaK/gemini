import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import wandb
from ygo.models.command_request import CommandRequest

from src.agents.hierarchical.models.action_scorer import ActionScorer
from src.agents.hierarchical.models.cnn import HierarchicalCNN
import src.agents.hierarchical.params as params
from src.agents.hierarchical.training.dataset import create_hierarchical_datasets, hierarchical_collate_fn
from src.agents.hierarchical.training.loss import HierarchicalLoss
import src.config as config
from src.feature.feature_manager import FeatureManager


def init_seed(seed: int) -> None:
    """
    シードを固定する。

    Args:
        seed (int): シード値
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment(args: argparse.Namespace) -> tuple[Path, torch.device]:
    """
    実験のセットアップを行う。

    Args:
        args (argparse.Namespace): コマンドライン引数

    Returns:
        tuple[Path, torch.device]: (保存ディレクトリ, デバイス)
    """
    # シード固定
    init_seed(params.SEED)

    # 保存ディレクトリ作成
    save_dir: Path = config.MODELS_DIR / args.group / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # パラメータバックアップ
    shutil.copy(params.__file__, save_dir / "params.py")

    # デバイス設定
    device: torch.device = torch.device(args.device)

    # WandB初期化
    wandb.init(
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
        group=args.group,
        name=args.name,
        config={k: v for k, v in vars(params).items() if not k.startswith("__")},
    )

    # ログ出力
    print(f"WandB Entity    : {config.WANDB_ENTITY}")
    print(f"WandB Project   : {config.WANDB_PROJECT}")
    print(f"WandB Group     : {args.group}")
    print(f"WandB Name      : {args.name}")
    print(f"Save Directory  : {save_dir}")
    print(f"Using Device    : {device}")
    print(f"Random Seed     : {params.SEED}")

    return save_dir, device


def prepare_data() -> tuple[DataLoader, DataLoader]:
    """
    データ関連の準備を行う。

    Returns:
        tuple[DataLoader, DataLoader]: (学習用データローダー, 検証用データローダー)
    """
    # 特徴量マネージャー作成
    feature_manager: FeatureManager = FeatureManager(scaling_factor=params.SCALING_FACTOR)

    # データセット作成
    train_dataset, valid_dataset = create_hierarchical_datasets(
        feature_manager=feature_manager,
        valid_ratio=params.VALID_RATIO,
    )

    # データローダー作成
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=max((os.cpu_count() or 1) // 2, 1),
        pin_memory=True,
        collate_fn=hierarchical_collate_fn,
    )
    valid_loader: DataLoader = DataLoader(
        valid_dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=False,
        num_workers=max((os.cpu_count() or 1) // 2, 1),
        pin_memory=True,
        collate_fn=hierarchical_collate_fn,
    )

    # ログ出力
    print(f"Batch Size      : {params.BATCH_SIZE}")
    print(f"Train Samples   : {len(train_dataset)}")
    print(f"Valid Samples   : {len(valid_dataset)}")

    # ログ記録
    wandb.config.update(
        {
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset),
        }
    )

    return train_loader, valid_loader


def build_model(device: torch.device) -> torch.nn.Module:
    """
    モデルを構築する。

    Args:
        device (torch.device): デバイス

    Returns:
        torch.nn.Module: モデル
    """
    # モデル構築
    model: torch.nn.Module = HierarchicalCNN(
        channels=config.TOTAL_CHANNELS_STATE_ACTION,
        image_size=(config.HEIGHT, config.WIDTH),
        num_block=params.NUM_BLOCKS,
        hidden_dim=params.HIDDEN_DIM,
        reduction_dim=params.REDUCTION_DIM,
        context_dim=params.CONTEXT_DIM,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dropout=params.DROPOUT,
    ).to(device)

    # モデル構造表示
    summary(
        model,
        input_size=(
            params.BATCH_SIZE,
            config.TOTAL_CHANNELS_STATE_ACTION,
            config.HEIGHT,
            config.WIDTH,
        ),
    )

    # ログ記録
    wandb.config.update(
        {
            "model_structure": str(model),
        }
    )

    return model


def _calculate_metrics(
    model_output: dict[str, torch.Tensor],
    command_requests: list[CommandRequest],
    correct_indices: list[int],
) -> tuple[int, int, int]:
    """
    正解数(Top-1, Top-2, Top-3)を計算する。

    Args:
        model_output (dict[str, torch.Tensor]): モデルの出力 (各Headのロジット)
        command_requests (list[CommandRequest]): コマンドリクエストのリスト
        correct_indices (list[int]): 正解インデックスリスト
    """
    correct_top1 = 0
    correct_top2 = 0
    correct_top3 = 0

    # スコア計算
    batch_action_scores = ActionScorer.calculate_scores(model_output, command_requests)

    for action_scores, correct_idx in zip(batch_action_scores, correct_indices):
        # Top-K 取得
        k = min(3, len(action_scores))
        topk_indices = torch.topk(action_scores, k=k).indices

        # Top-1
        if correct_idx in topk_indices[:1]:
            correct_top1 += 1

        # Top-2
        if correct_idx in topk_indices[:2]:
            correct_top2 += 1

        # Top-3
        if correct_idx in topk_indices[:3]:
            correct_top3 += 1

    return correct_top1, correct_top2, correct_top3


def train_step(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    1エポック分の訓練を行う。

    Args:
        model (torch.nn.Module): モデル
        loss_function (torch.nn.Module): 損失関数
        optimizer (torch.optim.Optimizer): オプティマイザ
        loader (DataLoader): データローダー
        device (torch.device): デバイス

    Returns:
        float: (平均損失, Top-1正解率, Top-2正解率, Top-3正解率)
    """
    # 訓練モード
    model.train()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top2 = 0
    total_correct_top3 = 0
    total_samples = 0

    for states, actions, command_requests, correct_indices in loader:
        # データをデバイスへ転送
        states = states.to(device)
        actions = {k: v.to(device) for k, v in actions.items()}
        batch_size = states.size(0)

        # 勾配初期化
        optimizer.zero_grad()

        # 推論
        outputs = model(states)

        # 損失計算
        loss = loss_function(outputs, actions)

        # 勾配更新
        loss.backward()
        optimizer.step()

        # 集計
        total_loss += loss.item() * batch_size
        correct_top1, correct_top2, correct_top3 = _calculate_metrics(
            model_output=outputs,
            command_requests=command_requests,
            correct_indices=correct_indices,
        )
        total_correct_top1 += correct_top1
        total_correct_top2 += correct_top2
        total_correct_top3 += correct_top3
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct_top1 / total_samples,
        total_correct_top2 / total_samples,
        total_correct_top3 / total_samples,
    )


def valid_step(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    1エポック分の検証を行う。

    Args:
        model (torch.nn.Module): モデル
        loss_function (torch.nn.Module): 損失関数
        loader (DataLoader): データローダー
        device (torch.device): デバイス

    Returns:
        float: (平均損失, Top-1正解率, Top-2正解率, Top-3正解率)
    """
    # 評価モード
    model.eval()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top2 = 0
    total_correct_top3 = 0
    total_samples = 0

    with torch.no_grad():
        for states, actions, command_requests, correct_indices in loader:
            # データをデバイスへ転送
            states = states.to(device)
            actions = {k: v.to(device) for k, v in actions.items()}
            batch_size = states.size(0)

            # 推論
            outputs = model(states)

            # 損失計算
            loss = loss_function(outputs, actions)

            # 集計
            total_loss += loss.item() * batch_size
            correct_top1, correct_top2, correct_top3 = _calculate_metrics(
                model_output=outputs,
                command_requests=command_requests,
                correct_indices=correct_indices,
            )
            total_correct_top1 += correct_top1
            total_correct_top2 += correct_top2
            total_correct_top3 += correct_top3
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct_top1 / total_samples,
        total_correct_top2 / total_samples,
        total_correct_top3 / total_samples,
    )


def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
) -> None:
    """
    学習を実行する。

    Args:
        model (torch.nn.Module): モデル
        train_loader (DataLoader): 学習用データローダー
        valid_loader (DataLoader): 検証用データローダー
        device (torch.device): デバイス
        save_dir (Path): 保存ディレクトリ
    """
    # 損失関数
    loss_function: torch.nn.Module = HierarchicalLoss()

    # オプティマイザ
    optimizer: torch.optim.Optimizer = optim.Adam(
        model.parameters(), lr=params.LEARNING_RATE, weight_decay=params.WEIGHT_DECAY
    )

    # 学習ループ
    for epoch in range(1, params.NUM_EPOCHS + 1):
        # 訓練
        train_loss, train_accuracy_top1, train_accuracy_top2, train_accuracy_top3 = train_step(
            model, loss_function, optimizer, train_loader, device
        )

        # 検証
        valid_loss, valid_accuracy_top1, valid_accuracy_top2, valid_accuracy_top3 = valid_step(
            model, loss_function, valid_loader, device
        )

        # ログ出力
        print(
            f"Epoch {epoch}/{params.NUM_EPOCHS}\n"
            f"  Train | Loss: {train_loss:.4f}   Accuracy: {train_accuracy_top1:.4f}\n"
            f"  Valid | Loss: {valid_loss:.4f}   Accuracy: {valid_accuracy_top1:.4f}"
        )

        # ログ記録
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy_top1": train_accuracy_top1,
                "train_accuracy_top2": train_accuracy_top2,
                "train_accuracy_top3": train_accuracy_top3,
                "valid_loss": valid_loss,
                "valid_accuracy_top1": valid_accuracy_top1,
                "valid_accuracy_top2": valid_accuracy_top2,
                "valid_accuracy_top3": valid_accuracy_top3,
            }
        )

        # モデル保存
        torch.save(model.state_dict(), save_dir / f"epoch_{epoch:03d}.pth")


def main() -> None:
    """
    メイン関数
    """
    # コマンドライン引数解析
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group", type=str, default="hierarchical-agent_sl")
    parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args: argparse.Namespace = parser.parse_args()

    # 1. 実験セットアップ
    print("★★★ Setting up Experiment ★★★")
    save_dir, device = setup_experiment(args)

    # 2. データ準備
    print("★★★ Preparing Data ★★★")
    train_loader, valid_loader = prepare_data()

    # 3. モデル構築
    print("★★★ Building Model ★★★")
    model = build_model(device)

    # 4. 学習実行
    print("★★★ Running Training ★★★")
    run_training(model, train_loader, valid_loader, device, save_dir)

    # WandB終了
    wandb.finish()


if __name__ == "__main__":
    main()
