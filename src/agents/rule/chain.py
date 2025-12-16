from typing import List, Optional

from ygo.models.command_request import CommandEntry
from ygo.models.duel_card import DuelCard
from ygo import constants as c

from src.env.state_data import StateData

# カードID定義
ID_CYCLONE = 1010
ID_HARPIE = 1007
ID_PREMATURE_BURIAL = 1009
ID_MONSTER_REBORN = 1025
ID_CALL_OF_HAUNTED = 1031
ID_RETURN_TO_FRONT = 1030
ID_POWERFUL_REBIRTH = 1016
ID_SILVERS_CRY = 1012
ID_MIRROR_FORCE = 1013
ID_TORRENTIAL_TRIBUTE = 1015
ID_DUST_TORNADO = 1014
ID_BOOK_OF_MOON = 1028
ID_LANCE = 1029
ID_SHRINK = 1011
ID_LIGHTNING_VORTEX = 1008
ID_BLUE_EYES = 1004
ID_PALADIN = 1024
ID_GENESIS_DRAGON = 1023
ID_ASSAULT_WYVERN = 1005
ID_ALEXANDRITE = 1003
ID_SAPPHIRE = 1002
ID_CAVE_DRAGON = 1001


class ChainExecutor:
    """
    チェーン処理および誘発即時効果のロジック (chain.txt準拠)
    相手のアクションに対するリアクションを担当
    """

    def __init__(self) -> None:
        pass

    def select_action(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """
        チェーン・誘発の行動選択
        """
        # チェーン確認の "DECIDE" (Yes/No) が来た場合
        # 具体的な発動コマンド (ACTIVATE) があればそちらを優先評価し、
        # なければ (または発動すべきでなければ) No/Pass を選ぶ形にする。
        
        # ACTIVATEコマンドがあるか確認
        activate_cmds = [cmd for cmd in commands if cmd.command_type == c.CommandType.ACTIVATE]
        
        if not activate_cmds:
            # 発動できるものがないなら、DECIDEで「いいえ(1)」などを返す
            # (agent.py の fallback に任せても良いが、ここで明示的に返す)
            return self._get_pass_or_no(commands)

        # 文脈判断: 「何に対して」チェーンしようとしているのか？
        # DuelStateData の chain_stack を確認
        duel_state = state.duel_state_data
        chain_stack = duel_state.chain_stack
        
        # チェーンスタックがある場合 (CL1以降)
        if len(chain_stack) > 0:
            last_chain = chain_stack[-1]
            # 相手の行動に対する反応か？
            if last_chain.player_id == c.PlayerId.RIVAL:
                return self._respond_to_opponent_chain(state, commands, last_chain.card_id)
            else:
                # 自分の行動へのさらなる追撃 (chain.txtにはあまり記述がないが、ダメステ等で発生)
                return self._respond_to_my_chain(state, commands)

        # チェーンスタックがない場合 (CL1のタイミング、攻撃宣言時、召喚成功時など)
        # ログなどから直前のイベントを推測する必要がある
        return self._respond_to_event(state, commands)

    # =========================================================================
    # A. 相手カード別対処 (chain.txt 準拠)
    # =========================================================================

    def _respond_to_opponent_chain(self, state: StateData, commands: List[CommandEntry], opp_card_id: int) -> Optional[CommandEntry]:
        """相手のチェーンブロックに対するカウンター判断"""
        
        # 1. ミラーフォース (Mirror Force) への対処
        if opp_card_id == ID_MIRROR_FORCE:
            # chain.txt: 主力に聖槍 or 月書
            cmd = self._logic_protect_ace(state, commands)
            if cmd: return cmd

        # 2. 激流葬 (Torrential Tribute) への対処
        elif opp_card_id == ID_TORRENTIAL_TRIBUTE:
            # chain.txt: 主力に聖槍
            cmd = self._logic_protect_ace(state, commands, use_book=False) # 月書では防げない(破壊される)が、裏で残るならあり？ txtは聖槍推奨
            if cmd: return cmd
            # 銀龍でリカバリ予約 (チェーン発動できるなら)
            cmd = self._find_command_by_card_id(commands, ID_SILVERS_CRY)
            if cmd: return cmd

        # 3. サイクロン / 砂塵 / 大嵐 (バック除去) への対処
        elif opp_card_id in [ID_CYCLONE, ID_DUST_TORNADO, ID_HARPIE]:
            # chain.txt: 割られるカードがリビデ/強化蘇生なら何かする、戦線復帰なら放置
            # 対象が何かわからないと正確な判断ができないが、
            # 「除去にチェーンして、損失を回避できる行動」があれば取る
            
            # 銀龍の轟咆 (墓地蘇生して手数を減らさない)
            cmd = self._find_command_by_card_id(commands, ID_SILVERS_CRY)
            if cmd: return cmd
            
            # 砂塵の大竜巻 (大嵐に対し、相手の永続を道連れにする)
            if opp_card_id == ID_HARPIE:
                cmd = self._find_command_by_card_id(commands, ID_DUST_TORNADO)
                if cmd: return cmd
                
            # 戦線復帰 (割られる前に使う)
            cmd = self._find_command_by_card_id(commands, ID_RETURN_TO_FRONT)
            if cmd: return cmd

        # 4. 早すぎた埋葬 / リビデ (蘇生) への対処
        elif opp_card_id in [ID_PREMATURE_BURIAL, ID_CALL_OF_HAUNTED]:
            # chain.txt: サイクロン/砂塵で無効化(不発化)
            cmd = self._find_command_by_card_id(commands, ID_CYCLONE)
            if cmd: return cmd
            cmd = self._find_command_by_card_id(commands, ID_DUST_TORNADO)
            if cmd: return cmd

        # 5. 月の書 / 収縮 / 聖槍 (対象を取る効果) への対処
        elif opp_card_id in [ID_BOOK_OF_MOON, ID_SHRINK, ID_LANCE]:
            # chain.txt: 聖槍で無効化、月書で回避(対象ずらし)
            # 聖槍があれば撃つ
            cmd = self._find_command_by_card_id(commands, ID_LANCE)
            if cmd: return cmd
            # 月の書で回避 (対象になった自分のモンスターを裏返す等)
            cmd = self._find_command_by_card_id(commands, ID_BOOK_OF_MOON)
            if cmd: return cmd

        # 6. ライトニング・ボルテックス
        elif opp_card_id == ID_LIGHTNING_VORTEX:
            cmd = self._logic_protect_ace(state, commands)
            if cmd: return cmd

        return self._get_pass_or_no(commands)

    # =========================================================================
    # C. ダメージステップ / 戦闘関連 (chain.txt 準拠)
    # =========================================================================

    def _respond_to_my_chain(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """自分の行動に更にチェーンするか（主にダメステ）"""
        # 基本的には相手の行動を待つが、ダメステで「収縮」→「聖槍」のような重ね掛けが必要な場合
        # 現状のシンプルな実装ではパス
        return self._get_pass_or_no(commands)

    def _respond_to_event(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """
        チェーンスタックが空＝イベント発生直後のタイミング (CL1)
        攻撃宣言時、召喚成功時、フェイズ移行時など
        """
        duel_state = state.duel_state_data
        
        # --- 1. 攻撃宣言時 (Attack Declaration) ---
        # ログの最後が「攻撃」であれば反応
        # ミラーフォース
        mf_cmd = self._find_command_by_card_id(commands, ID_MIRROR_FORCE)
        if mf_cmd:
            # 相手の攻撃モンスター数などを確認したい (off.py とロジック重複あり)
            # ここでは「撃てるなら撃つ」か、off.pyのロジックに任せるべきだが、
            # ChainExecutorが呼ばれている以上、ここで決める
            return mf_cmd

        # --- 2. 召喚・特殊召喚成功時 (Summon Success) ---
        # 激流葬
        tt_cmd = self._find_command_by_card_id(commands, ID_TORRENTIAL_TRIBUTE)
        if tt_cmd:
            # chain.txt: "召喚_特殊召喚_直後: 激流葬_発動"
            # 相手フィールドの状況を見て判断
            opp_monsters = self._get_opponent_monsters(duel_state)
            if len(opp_monsters) >= 2 or any(m.atk_val >= 2000 for m in opp_monsters):
                return tt_cmd

        # --- 3. ダメージステップ (Damage Step) ---
        # 収縮、聖槍
        if duel_state.general_data.current_phase == c.Phase.BATTLE:
             # ダメステ判定が難しいが、収縮がコマンドにある＝ダメステの可能性大
             shrink_cmd = self._find_command_by_card_id(commands, ID_SHRINK)
             lance_cmd = self._find_command_by_card_id(commands, ID_LANCE)
             
             # 自分のモンスターが戦闘中なら補助
             if shrink_cmd or lance_cmd:
                 # 戦闘ログ解析が必要だが、簡易的に「コマンドがあるなら使う」
                 # ただし無駄打ちは避けるため、自分のモンスターがいる場合のみ
                 if self._get_my_monsters(duel_state):
                     return shrink_cmd if shrink_cmd else lance_cmd

        return self._get_pass_or_no(commands)

    # =========================================================================
    # ヘルパーロジック
    # =========================================================================

    def _logic_protect_ace(self, state: StateData, commands: List[CommandEntry], use_book: bool = True) -> Optional[CommandEntry]:
        """主力を守るためのチェーン (聖槍、月の書)"""
        # 1. 禁じられた聖槍 (耐性付与)
        lance = self._find_command_by_card_id(commands, ID_LANCE)
        if lance: return lance
        
        # 2. 月の書 (裏守備にして回避)
        if use_book:
            book = self._find_command_by_card_id(commands, ID_BOOK_OF_MOON)
            if book: return book
            
        return None

    def _get_opponent_monsters(self, duel_state: StateData) -> List[DuelCard]:
        return [c for c in duel_state.duel_card_table 
                if c.player_id == c.PlayerId.RIVAL and c.position == c.PosId.MZONE]

    def _get_my_monsters(self, duel_state: StateData) -> List[DuelCard]:
        return [c for c in duel_state.duel_card_table 
                if c.player_id == c.PlayerId.MYSELF and c.position == c.PosId.MZONE]

    def _find_command_by_card_id(self, commands: List[CommandEntry], card_id: int) -> Optional[CommandEntry]:
        for cmd in commands:
            if cmd.command_type == c.CommandType.ACTIVATE:
                if cmd.card_id == card_id:
                    return cmd
        return None

    def _get_pass_or_no(self, commands: List[CommandEntry]) -> CommandEntry:
        """チェーンしない / いいえ / パス を選ぶ"""
        # DECIDEの場合: 通常 index 1 が No
        decide_cmds = [cmd for cmd in commands if cmd.command_type == c.CommandType.DECIDE]
        if decide_cmds:
            if len(decide_cmds) > 1:
                return decide_cmds[1]
            return decide_cmds[0] # 選択肢1つならそれしかない
            
        # PASS
        for cmd in commands:
            if cmd.command_type == c.CommandType.PASS:
                return cmd
        
        # なければ先頭
        return commands[0]