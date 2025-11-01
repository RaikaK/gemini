from src.env.action_data import ActionData
from src.env.state_data import StateData
from ygo.util.text import TextUtil


SYSTEM_PROMPT = """
あなたは、遊技王オフィシャルカードゲームでブルーアイズホワイトドラゴンを主軸としたデッキで世界一のプレイヤーです。
"""

NORMAL_RULE = """- 各プレイヤーのライフポイント(LP)は8000から始まる。
- 勝利条件は、相手のLPを0にするか、相手がデッキからカードをドローできなくなることです。(つまり、あなたのLPが0になるか、デッキがなくなると負けます。)
- ターンの流れ: あなたは以下の流れに従って遊技王をプレイします。
    * ドローフェーズ: デッキからカードを1枚ドローします。
    * スタンバイフェイズ: 特定のカードの効果やコストの支払いがここで行われることがあります。例えば、「スタンバイフェイズに発動する」効果を持つカードが場に出ている場合、このタイミングで効果が発動します。特にアクションがない場合は、このフェイズはそのままスキップします。
    * メインフェイズ1: モンスターの通常召喚、魔法・罠カードの発動を行うことができます。通常召喚は1ターンに1回だけ行えますが、特殊召喚はカード効果により何度でも行うことができます。魔法カード使用や罠カードのセットはターン中何度でも使用することができます。罠カードはセットしたターン中は発動できず、次の相手のターンから使用することができます。
    * バトルフェイズ: このフェイズで、プレイヤーはモンスターを使って相手モンスターに攻撃したり、相手プレイヤーに直接攻撃を仕掛け相手のLPを削ることができます。モンスターごとに攻撃力が異なり、攻撃の結果によって相手のライフポイントが減ったり、モンスターが破壊されたりします。
    * メインフェイズ2: バトルフェイズが終了した後、メインフェイズ2に移行します。ここでは、メインフェイズ1と同様に、モンスターの召喚や魔法・罠カードの発動が可能です。バトル後の状況に応じて、追加の戦略を練ることができます。(ただし、メインフェイズ1でモンスターカードの通常召喚をしていた場合は、モンスターカードを通常することはできません)
    * エンドフェイズ: ターン終了を宣言するフェーズです。エンドフェイズに発動する効果を持つカードがある場合、その処理を行います。このフェイズが終わると、相手のターンに移行します。
- モンスターカードの通常召喚/セット:
    * モンスターカードにはレベルが存在します。
    * レベル4以下のカード: リリースなしで召喚/セットが可能
    * レベル5・6: モンスターを1体リリースして、召喚します。
    * レベル7以上: モンスターを2体リリースして、召喚します。
    * リリースとは: リリースされたモンスターは、生贄として扱われ、バトルフィールドから排除されます。
- カードの種類:
    * モンスターカード: フィールドで戦闘に参加するカード。モンスターカードには特殊な能力や攻撃力、守備力が存在します。
    * 魔法カード: 基本的には手札から発動し、硬化処理後に墓地へ送られます。
    * 永続魔法カード: 
    * 罠カード: フィールドにセットすることで、次のターン以降に効果を発動できる。"""


class PromptGenerator:
    def __init__(self):
        self.text_util = TextUtil()

    def generate_instruction_prompt(self, state: StateData) -> str:
        prompt = f"""以下の**遊技王カードゲーム**において、ゲームに勝つことを目的に最適な選択肢を<Commands>の中から1つ数字で選びなさい。ただし、ゲームのルールは、<Rule>、盤面上のカードに関する情報は、<Card Table>、これまでのゲームの流れは<Log>、相手との呪文の打ち合い(チェイン)に関する情報は、<Chain Stack>、ライフポイント(LP)やターン情報などの一般情報は、<General Data>に記載されています。<Instruction>の指示にしたがい、最適な選択肢を指定されたフォーマットで出力してください。

<Rule>
{NORMAL_RULE}
</Rule>

<Card Table>
{self._get_duel_card_table_text(state=state)}
</Card Table>

<Log>
{self._get_log_text(state=state)}
</Log>

<Chain Stack>
{self._get_chain_stack_text(state=state)}
</Chain Stack>

<General Data>
{self._get_general_data_text(state=state)}
</General Data>

<Commands>
{self._get_command_list_text(state=state)}
</Commands>

<Instruction>
- あなたの思考の過程を"reasoning"として300文字程度で出力すること。
- <Commands>に記載されてある数字を1つ選び行動を選択すること
- 行動選択の前に必ず"reasoning"を出力すること
- 以下のJSONフォーマットで出力すること
{{
    "reasoning": "<あなたの思考の過程>",
    "action": <選択した数字>
}}
</Instruction>

あなたが現在取るべき最適な行動を選択しなさい。"""
        return prompt

    def _get_duel_card_table_text(self, state: StateData) -> str:
        duel_card_table = state.duel_state_data.duel_card_table
        return self.text_util.get_duel_card_table_markdown(
            duel_card_table=duel_card_table
        )

    def _get_command_list_text(self, state: StateData) -> str:
        """選択可能なコマンドリスト"""
        command_request = state.command_request
        return self.text_util.get_commands_text(commands=command_request.commands)

    def _get_log_text(self, state: StateData) -> str:
        """デュエルログ"""
        duel_log_entries = state.duel_log_data
        log_text = ""
        for duel_log_entry in duel_log_entries:
            log_text += (
                f"- {self.text_util.get_duel_log_entry_text([duel_log_entry])}\n"
            )
        return log_text

    def _get_chain_stack_text(self, state: StateData) -> str:
        """チェインスタック"""
        chain_stack = state.duel_state_data.chain_stack
        return self.text_util.get_chain_stack_text(chain_stack=chain_stack)

    def _get_general_data_text(self, state: StateData) -> str:
        """一般情報"""
        duel_state = state.duel_state_data
        return self.text_util.get_general_data_text(duel_state=duel_state)
