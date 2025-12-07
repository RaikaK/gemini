import torch
import torch.nn as nn

from src.agents.hierarchical.models.action_heads import ActionHeads


class ResNetBlock(nn.Module):
    """
    ResNet Block
    """

    def __init__(
        self,
        *,
        dim: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        """
        初期化する。

        Args:
            dim (int): チャンネル数
            kernel_size (tuple[int, int]): カーネルサイズ
            stride (tuple[int, int]): ストライド
            padding (tuple[int, int]): パディング

        Attributes:
            conv1 (nn.Sequential): 1つ目の畳み込み層
            conv2 (nn.Sequential): 2つ目の畳み込み層
            relu (nn.ReLU): ReLU活性化関数
        """
        super().__init__()

        self.conv1: nn.Sequential = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.conv2: nn.Sequential = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            torch.Tensor: 出力テンソル
        """
        out: torch.Tensor = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(x + out)

        return out


class HierarchicalCNN(nn.Module):
    """
    階層型CNN
    """

    def __init__(
        self,
        *,
        channels: int,
        image_size: tuple[int, int],
        num_block: int,
        hidden_dim: int,
        reduction_dim: int,
        context_dim: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dropout: float,
    ) -> None:
        """
        初期化する。

        Args:
            channels (int): チャンネル数
            image_size (tuple[int, int]): 画像サイズ (H, W)
            num_block (int): ResNetブロック数
            hidden_dim (int): 隠れ層の次元数
            reduction_dim (int): 圧縮層の次元数
            context_dim (int): コンテキストベクトルの次元数
            kernel_size (tuple[int, int]): カーネルサイズ
            stride (tuple[int, int]): ストライド
            padding (tuple[int, int]): パディング
            dropout (float): ドロップアウト率

        Attributes:
            channels (int): チャンネル数
            image_height (int): 画像の高さ
            image_width (int): 画像の幅
            backbone_conv (nn.Sequential): 最初の畳み込み層
            resnet_blocks (nn.Sequential): ResNetブロック群
            reduction_conv (nn.Sequential): 次元圧縮層
            context_layer (nn.Sequential): コンテキスト生成層
            heads (nn.ModuleDict): 各Headの出力層
        """
        super().__init__()

        self.channels: int = channels
        self.image_height, self.image_width = image_size

        # --- Backbone ---
        self.backbone_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.resnet_blocks: nn.Sequential = nn.Sequential(
            *[
                ResNetBlock(
                    dim=hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
                for _ in range(num_block)
            ]
        )
        self.reduction_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(hidden_dim, reduction_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
        )
        self.context_layer: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduction_dim * self.image_height * self.image_width, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Heads ---
        self.heads: nn.ModuleDict = nn.ModuleDict(
            {
                action_head.name: nn.Linear(context_dim, action_head.output_dim)
                for action_head in ActionHeads.get_all_heads()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        順伝播を行う。

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            dict[str, torch.Tensor]: 各Headの出力テンソル
        """
        _, channels, height, width = x.shape

        assert (
            channels == self.channels
        ), f"Actual input channels ({channels}) must be equal to expected input channels ({self.channels})"
        assert (
            height == self.image_height
        ), f"Actual image height ({height}) must be equal to expected image height ({self.image_height})"
        assert (
            width == self.image_width
        ), f"Actual image width ({width}) must be equal to expected image width ({self.image_width})"

        # --- Backbone ---
        out: torch.Tensor = self.backbone_conv(x)
        out = self.resnet_blocks(out)
        out = self.reduction_conv(out)
        out = self.context_layer(out)

        # --- Heads ---
        out_dict: dict[str, torch.Tensor] = {name: layer(out) for name, layer in self.heads.items()}

        return out_dict
