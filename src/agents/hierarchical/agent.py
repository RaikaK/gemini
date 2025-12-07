import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from ygo.constants.enums import SelectionType
from ygo.models.command_request import CommandEntry, CommandRequest

from src.agents.base_agent import BaseAgent
from src.agents.hierarchical.models.action_scorer import ActionScorer
from src.agents.hierarchical.models.cnn import HierarchicalCNN
import src.config as config
from src.env.action_data import ActionData
from src.env.state_data import StateData
from src.feature.feature_manager import FeatureManager


class HierarchicalAgent(BaseAgent):
    """
    階層型エージェント
    """

    def __init__(self, model_path: Path) -> None:
        """
        初期化する。

        Args:
            model_path (Path): 学習済みモデルのパス

        Attributes:
            device (torch.device): デバイス
            model (torch.nn.Module): モデル
            feature_manager (FeatureManager): 特徴量マネージャー
        """
        # デバイス設定
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # パラメータ読み込み
        params: ModuleType = self._load_params(model_path)

        # モデル構築
        self.model: torch.nn.Module = HierarchicalCNN(
            channels=config.TOTAL_CHANNELS_STATE,
            image_size=(config.HEIGHT, config.WIDTH),
            num_block=params.NUM_BLOCKS,
            hidden_dim=params.HIDDEN_DIM,
            reduction_dim=params.REDUCTION_DIM,
            context_dim=params.CONTEXT_DIM,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dropout=params.DROPOUT,
        ).to(self.device)

        # モデルロード
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # 特徴量マネージャー作成
        self.feature_manager: FeatureManager = FeatureManager(scaling_factor=params.SCALING_FACTOR)

    def _load_params(self, model_path: Path) -> ModuleType:
        """
        パラメータを読み込む。

        Args:
            model_path (Path): モデルファイルのパス

        Returns:
            ModuleType: パラメータモジュール
        """
        params_path: Path = model_path.parent / config.PARAMS_FILE

        if not params_path.exists():
            raise FileNotFoundError(f"Params file not found at: {params_path}")

        spec = importlib.util.spec_from_file_location("params", params_path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load params module from: {params_path}")

        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)

        return params

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        command_request: CommandRequest = state.command_request
        selectable_commands: list[CommandEntry] = command_request.commands

        # --- ルールベース (ドロー) ---
        if command_request.selection_type == SelectionType.DRAW_PHASE:
            return ActionData(command_request, selectable_commands[0]), None

        # --- 推論ベース ---
        # 評価モード
        self.model.eval()

        # 1. 特徴量作成
        feature: np.ndarray = self.feature_manager.to_snapshot_policy_feature(state)
        feature_tensor: torch.Tensor = torch.from_numpy(feature).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 2. 推論
            outputs: dict[str, torch.Tensor] = self.model(feature_tensor)

            # 3. スコア計算
            action_scores: torch.Tensor = ActionScorer.calculate_scores(outputs, [command_request])[0]

        # 4. 行動選択
        best_command_idx: int = int(torch.argmax(action_scores).item())
        selected_command: CommandEntry = selectable_commands[best_command_idx]

        return ActionData(command_request, selected_command), None

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        return None
