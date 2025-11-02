import torch

"""教師あり学習モジュール"""


class SuperviseidLearning:
    """教師あり学習の基本クラス"""

    def __init__(self, model: torch.nn.Module):
        self.model: torch.nn.Module = model

    def train(self, X_train, y_train):
        """モデルの訓練を行う"""
        pass
