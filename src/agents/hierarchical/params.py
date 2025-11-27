import os
import numpy as np
import random
import torch

# =============================================================================
# ハイパーパラメータ設定
# =============================================================================

# シード値
SEED = 4007

# =============================================================================
# シード固定ロジック
# =============================================================================


def init_seed(seed: int) -> None:
    """
    シードを固定する。

    Args:
        seed (int): シード値
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
