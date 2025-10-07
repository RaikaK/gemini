from ygo.models import DuelEndData
from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest, CommandEntry
from ygo.util.card import CardUtil
from ygo.constants import EffectNo


class CardInfo:
    def __init__(self):
        self.card_util = CardUtil()

    def get_card_info(self, card_id):
        # カードの名前
        name = self.card_util.get_name(card_id)

        # レベル
        card_level = self.card_util.get_level(card_id)

        # 攻撃力と守備力
        atk = self.card_util.get_atk(card_id=card_id)
        deff = self.card_util.get_def(card_id=card_id)

        # 能力
        effect_text = "\n".join(
            [
                self.card_util.get_effect_text(card_id, effect_no)
                for effect_no in range(EffectNo.NUM1, EffectNo.NUM5 + 1)
                if self.card_util.get_effect_text(card_id, effect_no) != ""
            ]
        )

        info = f"[{name}]\n- レベル: {card_level}\n- 効果:\n{effect_text}\n攻撃力={atk} / 守備力={deff}"
        return info


if __name__ == "__main__":
    card_info = CardInfo()
    card_id = 1018  # ボマードラゴン
    print(card_info.get_card_info(card_id))
