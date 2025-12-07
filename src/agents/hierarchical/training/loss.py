import torch
import torch.nn as nn

from src.agents.hierarchical.models.action_heads import ActionHeads


class HierarchicalLoss(nn.Module):
    """
    階層型損失関数
    """

    def __init__(self) -> None:
        """
        初期化する。
        """
        super().__init__()

        # -1 (無効値) を無視して平均をとる設定
        self.loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        合計損失を計算する。

        Args:
            model_output (dict[str, torch.Tensor]): モデルの出力 (各Headのロジット)
            targets (dict[str, torch.Tensor]): 正解行動ラベル (各Headのインデックス)

        Returns:
            torch.Tensor: 合計損失
        """
        # デバイス取得
        device: torch.device = model_output[ActionHeads.COMMAND_TYPE.name].device
        total_loss: torch.Tensor = torch.tensor(0.0, device=device)

        # 合計損失を計算
        for action_head in ActionHeads.get_all_heads():
            head_name: str = action_head.name

            # 予測と正解を取得
            prediction: torch.Tensor = model_output[head_name]
            target: torch.Tensor = targets[head_name]

            # 各Headの損失を計算
            loss = self.loss_function(prediction, target)

            # 損失がNaNでない場合に加算
            if not torch.isnan(loss):
                total_loss += loss

        return total_loss
