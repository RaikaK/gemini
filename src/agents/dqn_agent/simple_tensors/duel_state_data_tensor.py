import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import torch

from src.agents.dqn_agent.simple_tensors.simple_tensor import simple_dataclass_tensor

from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest
from ygo.models.duel_card import DuelCard
from ygo.models.general_data import GeneralData
from ygo.models.chain_data import ChainData

NUM_DUEL_CARD_TABLE = 40  # 最大200らしい (by vendor/UDI/samples/basic/duel_data_sample.ipynb) 今回はブルーアイズデッキ飲みなので、40とする

DIM_DUEL_CARD = len(DuelCard.__dataclass_fields__)
DIM_GENERAL_DATA = len(GeneralData.__dataclass_fields__)
DIM_CHAIN_DATA = len(ChainData.__dataclass_fields__)
MAX_CHAIN_TARGET = (5 + 5) * 2  # モンスター5+魔法トラップゾーン5 * 自分と相手2
MAX_CHAIN_STACK = 20  # 最大20より大きくならないことを祈る

# 最終的なDuelStateDataのテンソルの次元数
DIM_DUEL_STATE_DATA = (
    DIM_DUEL_CARD * NUM_DUEL_CARD_TABLE * 2
    + DIM_GENERAL_DATA
    + (DIM_CHAIN_DATA + MAX_CHAIN_TARGET) * MAX_CHAIN_STACK
)


def simple_duel_state_data_tensor(
    duel_state_data: DuelStateData, dim: int = DIM_DUEL_STATE_DATA, dtype=torch.float32
) -> torch.Tensor:
    """
    DuelStateDataをテンソル化する
    """
    # duel_card_table
    duel_card_table: list[DuelCard] = duel_state_data.duel_card_table
    duel_card_table_tensor = simple_duel_card_table_tensor(
        duel_card_table=duel_card_table
    )

    # general_data
    general_data: GeneralData = duel_state_data.general_data
    general_data_tensor = simple_dataclass_tensor(general_data, dim=-1)

    # chain_stack
    chain_stack: list[ChainData] = duel_state_data.chain_stack
    chain_stack_tensor: torch.Tensor = simple_chain_stack_tensor(
        chain_stack=chain_stack,
        dim_chain_data=(DIM_CHAIN_DATA + MAX_CHAIN_TARGET),
        max_chain_stack=MAX_CHAIN_STACK,
        dtype=dtype,
    )

    # 最終的なテンソルを作成
    duel_state_data_tensor = torch.cat(
        [
            duel_card_table_tensor,
            general_data_tensor,
            chain_stack_tensor,
        ],
        dim=0,
    )
    if len(duel_state_data_tensor) > dim:
        return duel_state_data_tensor[:dim]
    elif len(duel_state_data_tensor) < dim:
        print("pad duel_state_data_tensor to dim:", dim)
        padded_tensor = torch.zeros(dim, dtype=dtype)
        padded_tensor[: len(duel_state_data_tensor)] = duel_state_data_tensor
        return padded_tensor
    return duel_state_data_tensor


def simple_chain_stack_tensor(
    chain_stack: list[ChainData],
    dim_chain_data: int = (DIM_CHAIN_DATA + MAX_CHAIN_TARGET),
    max_chain_stack: int = MAX_CHAIN_STACK,
    dtype=torch.float32,
) -> torch.Tensor:
    chain_stack_tensor_list = [
        simple_dataclass_tensor(chain_data, dim=dim_chain_data, dtype=dtype, slice=True)
        for chain_data in chain_stack
        if isinstance(chain_data, ChainData)
    ]
    # MAX_CHAIN_STACKを超えてしまった場合、スライスする
    if len(chain_stack_tensor_list) > max_chain_stack:
        return torch.cat(chain_stack_tensor_list[:max_chain_stack], dim=0)
    # 足りない分は0でパディング
    elif len(chain_stack_tensor_list) < max_chain_stack:
        zero_tensors = [
            torch.zeros(dim_chain_data, dtype=dtype)
            for i in range(max_chain_stack - len(chain_stack_tensor_list))
        ]
        chain_stack_tensor_list.extend(zero_tensors)[:max_chain_stack]
        return torch.cat(chain_stack_tensor_list, dim=0)
    return torch.cat(chain_stack_tensor_list, dim=0)


def simple_duel_card_table_tensor(
    duel_card_table: list[DuelCard],
    dim: int = NUM_DUEL_CARD_TABLE * DIM_DUEL_CARD,
    dtype=torch.float32,
) -> torch.Tensor:
    duel_card_tensor_list = [
        simple_dataclass_tensor(duel_card, dim=DIM_DUEL_CARD, dtype=dtype, slice=True)
        for duel_card in duel_card_table
        if isinstance(duel_card, DuelCard)
    ]
    duel_card_table_tensor = torch.cat(
        duel_card_tensor_list[:NUM_DUEL_CARD_TABLE]
        + duel_card_tensor_list[100 : 100 + NUM_DUEL_CARD_TABLE :]
    )
    return duel_card_table_tensor
