import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import torch
from dataclasses import dataclass


"""
YgoEnvの状態'state'はDuelStateDataであるが、
この要素には、
- duel_card_table: list[DuelCard]
- general_data: GeneralData
- chain_stack: list[ChainData]
の3つが含まれており、これらを全てテンソル化する
"""


def simple_dataclass_tensor(
    data, dim: int = -1, dtype=torch.float32, slice: bool = False
) -> torch.Tensor:
    """
    dataclassを単純にテンソル化する
    - dim=-1: dataclassの中身を全てフラットなテンソルとして返す
    - slice=True: dimが指定されている場合、dimにスライスして返す
    - その他の場合: 指定したdimの大きさで0でパディングして返す
    """
    if dim == 0:
        raise ValueError("dim must be -1 or positive integer")
    vals = []
    for field in data.__dataclass_fields__.keys():
        attr = getattr(data, field)
        if isinstance(attr, list):
            vals.extend(attr)
        elif isinstance(attr, bool):
            vals.append(1.0 if attr else 0.0)
        else:
            vals.append(attr)

    if dim == -1:
        return torch.tensor(vals, dtype=dtype)
    # valsが指定したdimより大きく、sliceがTrueならば、スライスして返す
    elif len(vals) > dim and slice:
        print("slice tensor to dim:", dim)
        return torch.tensor(vals[:dim], dtype=dtype)
    else:
        tensor = torch.zeros(dim, dtype=dtype)
        for i in range(min(len(vals), dim)):
            tensor[i] = vals[i]
        return tensor


if __name__ == "__main__":

    @dataclass
    class Sample:
        a: int
        b: int
        c: list[int]

    sample = Sample(a=1, b=2, c=[3, 14])

    sample_tensor = simple_dataclass_tensor(sample, dim=10, slice=True)
    print(sample_tensor)
    breakpoint()
