import torch
import torch.nn.functional as F

from ygo.models.command_request import CommandRequest

from src.agents.hierarchical.models.action_heads import ActionHeads


class ActionScorer:
    """
    アクションスコアラー
    """

    @classmethod
    def calculate_scores(
        cls,
        model_output: dict[str, torch.Tensor],
        command_requests: list[CommandRequest],
    ) -> list[torch.Tensor]:
        """
        コマンドリクエストの行動選択肢に対するスコアを計算する。

        Args:
            model_output (dict[str, torch.Tensor]): モデルの出力 (各Headのロジット)
            command_requests (list[CommandRequest]): コマンドリクエストのリスト

        Returns:
            list[torch.Tensor]: 行動選択肢に対するスコアのリスト
        """
        # デバイス取得
        device = model_output[ActionHeads.COMMAND_TYPE.name].device
        batch_action_scores: list[torch.Tensor] = []

        # 各HeadのLogSoftmaxを計算
        heads_log_probs = {k: F.log_softmax(v, dim=1) for k, v in model_output.items()}

        # 各コマンドリクエストのスコアを計算
        for i, command_request in enumerate(command_requests):
            action_scores: list[torch.Tensor] = []

            # 各行動選択肢に対するスコアを計算
            for command in command_request.commands:
                action_score = torch.tensor(0.0, device=device)

                # 各Headのスコアを加算
                for action_head in ActionHeads.get_all_heads():
                    val = getattr(command, action_head.name)
                    idx = ActionHeads.to_head_index(action_head.name, val)

                    if idx != -1:
                        action_score += heads_log_probs[action_head.name][i][idx]

                action_scores.append(action_score)

            if action_scores:
                batch_action_scores.append(torch.stack(action_scores))

            else:
                batch_action_scores.append(torch.tensor([], device=device))

        return batch_action_scores
