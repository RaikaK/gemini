import random
from typing import List, Optional, Dict, Any, Tuple

from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from ygo.models.duel_card import DuelCard
from ygo import constants as c
from ygo.util import card as card_util

from src.env.state_data import StateData
from src.env.action_data import ActionData

# カードID定義
ID_CYCLONE = 1010
ID_HARPIE = 1007  # 大嵐
ID_POT_OF_GREED = 1006
ID_LIGHTNING_VORTEX = 1008
ID_PREMATURE_BURIAL = 1009
ID_MONSTER_REBORN = 1025
ID_ADVANCED_RITUAL_ART = 1027
ID_WHITE_DRAGON_RITUAL = 1026
ID_PALADIN = 1024
ID_BLUE_EYES = 1004
ID_ASSAULT_WYVERN = 1005
ID_GENESIS_DRAGON = 1023
ID_MANJU = 1022
ID_SENJU = 1020
ID_SONIC_BIRD = 1021
ID_ALEXANDRITE = 1003
ID_SAPPHIRE = 1002
ID_CAVE_DRAGON = 1001
ID_MASKED_DRAGON = 1017
ID_BOMBER_DRAGON = 1018
ID_KIDMODO = 1019
ID_MIRROR_FORCE = 1013
ID_TORRENTIAL_TRIBUTE = 1015
ID_DUST_TORNADO = 1014
ID_CALL_OF_HAUNTED = 1031
ID_RETURN_TO_FRONT = 1030
ID_POWERFUL_REBIRTH = 1016
ID_SILVERS_CRY = 1012
ID_BOOK_OF_MOON = 1028
ID_LANCE = 1029
ID_SHRINK = 1011


class AtkExecutor:
    """
    自分ターンの行動ロジック (atk.txt準拠)
    """

    def __init__(self) -> None:
        pass

    def select_action(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """
        自分ターンの行動を選択するメインメソッド
        """
        duel_state = state.duel_state_data
        general = duel_state.general_data
        request = state.command_request
        selection_type = request.selection_type

        # 1. 選択タイプによる分岐
        if selection_type == c.SelectionType.COMMAND:
            # フェイズごとのコマンド選択
            if general.current_phase == c.Phase.DRAW:
                return self._phase_draw(commands)
            elif general.current_phase == c.Phase.STANDBY:
                return self._phase_standby(state, commands)
            elif general.current_phase == c.Phase.MAIN1:
                return self._phase_main1(state, commands)
            elif general.current_phase == c.Phase.BATTLE:
                return self._phase_battle(state, commands)
            elif general.current_phase == c.Phase.MAIN2:
                return self._phase_main2(state, commands)
            elif general.current_phase == c.Phase.END:
                return self._phase_end(state, commands)
            else:
                # デフォルト
                return commands[0]

        # 2. 対象選択・カード選択 (SELECT_CARD 等)
        # atk.txt にある「ティア表」や「コスト選択」ロジックを使用
        if selection_type in [c.SelectionType.SELECT_CARD, c.SelectionType.SELECT_UNSELECT_CARD]:
            return self._select_card_target(state, commands)
        
        # 3. 表示形式選択
        if selection_type == c.SelectionType.SELECT_POSITION:
            return self._select_position(state, commands)
            
        # 4. Yes/No 選択 (DECIDE)
        if selection_type == c.SelectionType.DECIDE:
            # 基本的にEffect発動確認など。atk.txtのロジックに従いYes/Noを決める
            # ここでは簡易的に、有利な効果ならYesとするが、本来はcontextが必要
            return commands[0]

        return None

    # =========================================================================
    # フェイズ別ロジック
    # =========================================================================

    def _phase_draw(self, commands: List[CommandEntry]) -> CommandEntry:
        """ドローフェイズ: 基本は何もしない（パス/フェイズ移行）"""
        # 特殊なフリーチェーン発動がない限りパス
        return self._get_priority_command(commands, [c.CommandType.PASS, c.CommandType.CHANGE_PHASE])

    def _phase_standby(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """スタンバイフェイズ: エンドサイク的な処理や維持コスト払いなど"""
        duel_state = state.duel_state_data
        # 条件: サイクロンセット済み and 相手伏せあり
        # ここではセット済みのサイクロンを発動できるかチェック
        
        # 相手のフィールド状況
        opponent_cards = self._get_opponent_field(duel_state)
        opp_set_cards = [c for c in opponent_cards if c.position in [c.PosId.MZONE, c.PosId.SZONE] and c.face == c.Face.DOWN]
        
        if len(opp_set_cards) >= 1:
            # サイクロン発動コマンドを探す
            cyclone_cmd = self._find_command_by_card_id(commands, ID_CYCLONE, command_type=c.CommandType.ACTIVATE)
            if cyclone_cmd:
                # 確率的要素や詳細条件（atk.txt準拠）
                # 相手モンスター数などの条件があればここでチェック
                return cyclone_cmd

        # 基本はフェイズ移行
        return self._get_priority_command(commands, [c.CommandType.CHANGE_PHASE, c.CommandType.PASS])

    def _phase_main1(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """メインフェイズ1: 展開・除去の主戦場"""
        duel_state = state.duel_state_data
        
        # --- 1. 魔法・罠による除去 (伏せ除去優先) ---
        cmd = self._logic_main_removal(state, commands)
        if cmd: return cmd

        # --- 2. 儀式召喚・特殊召喚の展開 ---
        cmd = self._logic_ritual_summon(state, commands)
        if cmd: return cmd
        
        # --- 3. 青眼召喚判断 (白竜の聖騎士の効果) ---
        cmd = self._logic_paladin_effect(state, commands)
        if cmd: return cmd

        # --- 4. 蘇生・召喚補助 ---
        cmd = self._logic_revive(state, commands)
        if cmd: return cmd

        # --- 5. ドローソース (強欲な壺) ---
        # atk.txt: サーチ完了後に撃つ記述があるが、ここでは簡易的に優先度低めで発動
        pot_cmd = self._find_command_by_card_id(commands, ID_POT_OF_GREED, command_type=c.CommandType.ACTIVATE)
        if pot_cmd: return pot_cmd

        # --- 6. ライトニング・ボルテックス ---
        cmd = self._logic_lightning_vortex(state, commands)
        if cmd: return cmd
        
        # --- 7. 通常召喚 ---
        cmd = self._logic_normal_summon(state, commands)
        if cmd: return cmd

        # --- 8. 表示形式変更 ---
        cmd = self._logic_position_change(state, commands)
        if cmd: return cmd

        # --- 9. バトルフェイズへの移行 ---
        # 攻撃可能なモンスターがいる、またはバトルを行いたい場合
        to_battle = self._find_command_by_type(commands, c.CommandType.CHANGE_PHASE)
        if to_battle and to_battle.phase == c.Phase.BATTLE:
            return to_battle

        # --- 10. メイン2/エンドへの移行 ---
        # バトルに行けない/行きたくない場合
        return self._get_priority_command(commands, [c.CommandType.CHANGE_PHASE, c.CommandType.PASS])

    # =========================================================================
    # ロジック詳細 (メインフェイズ)
    # =========================================================================

    def _logic_main_removal(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """サイクロン、大嵐などの伏せ除去"""
        duel_state = state.duel_state_data
        opp_field = self._get_opponent_field(duel_state)
        opp_backrow = [c for c in opp_field if c.position == c.PosId.SZONE]
        opp_monsters = [c for c in opp_field if c.position == c.PosId.MZONE]

        # サイクロン
        # atk.txt: 相手伏せ枚数 >= 1, 自分攻撃表示モンスター >= 1 or 召喚予定...
        if len(opp_backrow) >= 1:
            cyclone = self._find_command_by_card_id(commands, ID_CYCLONE, command_type=c.CommandType.ACTIVATE)
            if cyclone:
                # 条件チェック（厳密化）
                my_atk_monsters = [c for c in self._get_my_field(duel_state) if c.position == c.PosId.MZONE and c.face in [c.Face.FACEUP_ATTACK]]
                summon_plan = True # 簡易的に常に召喚予定ありとする
                
                # 相手モンスターが少ない、または青眼でないなら安全策で割る
                is_safe_target = len(opp_monsters) <= 1 or not any(c.card_id == ID_BLUE_EYES for c in opp_monsters)
                
                if (len(my_atk_monsters) >= 1 or summon_plan) and is_safe_target:
                    return cyclone

        # 大嵐
        # atk.txt: 自分の永続・伏せが少なく、相手の伏せが多い場合に発動
        if len(opp_backrow) >= 2: # 相手伏せ2枚以上なら検討
            harpie = self._find_command_by_card_id(commands, ID_HARPIE, command_type=c.CommandType.ACTIVATE)
            if harpie:
                my_backrow = [c for c in self._get_my_field(duel_state) if c.position == c.PosId.SZONE]
                # 自分の被害が少ない、またはこれから攻め切る場合
                if len(my_backrow) <= 1: 
                    return harpie
        
        return None

    def _logic_ritual_summon(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """儀式魔法の発動"""
        # 高等儀式術
        adv_ritual = self._find_command_by_card_id(commands, ID_ADVANCED_RITUAL_ART, command_type=c.CommandType.ACTIVATE)
        if adv_ritual:
            # 手札に白竜の聖騎士がいるかチェックが必要だが、発動可能コマンドに出ている時点で条件は満たしている可能性が高い
            return adv_ritual
            
        # 白竜降臨
        white_ritual = self._find_command_by_card_id(commands, ID_WHITE_DRAGON_RITUAL, command_type=c.CommandType.ACTIVATE)
        if white_ritual:
            return white_ritual
            
        return None

    def _logic_paladin_effect(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """白竜の聖騎士の効果②（リリースして青眼SS）"""
        # atk.txt: 
        # 1) 裏守備除去（①）を最大化したい：まず攻撃 → その後② が本線
        # 2) 相手に表側大型がいる：メイン2で
        # 3) ミラフォ/激流葬の危険：メイン1で（サクリファイスエスケープ的運用や、攻撃反応を踏まないため）
        
        paladin_eff = self._find_command_by_card_id(commands, ID_PALADIN, command_type=c.CommandType.ACTIVATE)
        if not paladin_eff:
            return None
            
        # 状況分析
        duel_state = state.duel_state_data
        opp_field = self._get_opponent_field(duel_state)
        opp_backrow = [c for c in opp_field if c.position == c.PosId.SZONE]
        opp_monsters = [c for c in opp_field if c.position == c.PosId.MZONE]
        opp_set_monsters = [c for c in opp_monsters if c.face == c.Face.DOWN_DEFENSE]
        
        has_threat_backrow = len(opp_backrow) >= 1
        has_faceup_strong = any(c.atk_val >= 1900 for c in opp_monsters if c.face != c.Face.DOWN_DEFENSE)
        
        do_activate = False
        
        # 1. 裏守備がいるなら、殴ってから効果を使いたい -> メイン1では使わない
        if len(opp_set_monsters) > 0:
            do_activate = False # バトルで殴る優先
            
            # ただし、激流葬/ミラフォ濃厚でケア手段がないならメイン1で変える選択肢
            # (簡易実装: 伏せが多くて守る手段がないなら変える)
            if len(opp_backrow) >= 2:
                do_activate = True
        
        # 2. 表側大型がいる -> 1900打点では勝てないので、メイン1ですぐ青眼に変えるか？
        # atk.txtには「殴ってから②」とあるが、勝てないなら意味がない。
        # 殴れる相手がいないなら変える
        elif has_faceup_strong and not any(c.atk_val < 1900 for c in opp_monsters if c.face != c.Face.DOWN_DEFENSE):
             do_activate = True
             
        # 3. 伏せが厚い -> メイン1で青眼（3000）にして圧力をかける、あるいは守備で出す？
        elif has_threat_backrow:
            # 安全重視ならメイン1
            do_activate = True
            
        # デフォルト: 殴れるなら殴ってメイン2
        
        if do_activate:
            return paladin_eff
        
        return None

    def _logic_revive(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """死者蘇生などの発動"""
        # atk.txt: 
        # キルレンジ、盤面復旧、守り固め、相手墓地奪取などの条件
        reborn = self._find_command_by_card_id(commands, ID_MONSTER_REBORN, command_type=c.CommandType.ACTIVATE)
        if not reborn:
            return None
            
        duel_state = state.duel_state_data
        my_field = self._get_my_field(duel_state)
        my_monsters = [c for c in my_field if c.position == c.PosId.MZONE]
        
        # 条件1: 盤面が空に近い (1体以下)
        if len(my_monsters) <= 1:
            return reborn
            
        # 条件2: 相手のLPが低い (4000以下) -> 攻め
        if duel_state.general_data.lp[c.PlayerId.RIVAL] <= 4000:
            return reborn
            
        return None

    def _logic_lightning_vortex(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """ライトニング・ボルテックス"""
        lv = self._find_command_by_card_id(commands, ID_LIGHTNING_VORTEX, command_type=c.CommandType.ACTIVATE)
        if not lv: return None
        
        duel_state = state.duel_state_data
        opp_field = self._get_opponent_field(duel_state)
        opp_faceup_monsters = [c for c in opp_field if c.position == c.PosId.MZONE and c.face != c.Face.DOWN_DEFENSE]
        
        # 価値評価: 破壊見込み枚数が2枚以上、またはキル圏内
        if len(opp_faceup_monsters) >= 2:
            return lv
        
        # 相手に高打点（青眼など）がいて突破困難な場合
        if any(c.atk_val >= 2000 for c in opp_faceup_monsters):
            return lv
            
        return None

    def _logic_normal_summon(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """通常召喚の選択"""
        # summon_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.SUMMON]
        # if not summon_commands: return None
        
        # atk.txt TIERロジック
        # TIER1: サーチ系 (マンジュ、センジュ、ソニバ)、高打点バニラ（アレキ、サファイア）
        # TIER2: 創世の竜騎士、アサルトワイバーン
        # TIER3: 仮面竜、ボマー、コドモ（受動的効果なので自ターン召喚は優先度低）
        
        # 手札にあるカードとコマンドのマッピングが必要
        # コマンドからカードIDを取得して判定
        
        # 優先順位リスト
        priority_ids = [
            # TIER 1
            ID_MANJU, ID_SENJU, ID_SONIC_BIRD,
            ID_ALEXANDRITE, ID_SAPPHIRE,
            # TIER 2
            ID_GENESIS_DRAGON, ID_ASSAULT_WYVERN,
            # TIER 3
            ID_MASKED_DRAGON, ID_BOMBER_DRAGON, ID_CAVE_DRAGON,
            ID_KIDMODO
        ]
        
        for card_id in priority_ids:
            cmd = self._find_command_by_card_id(commands, card_id, command_type=c.CommandType.SUMMON)
            if cmd:
                # 追加条件チェック
                # 例: 激流葬ケアで聖槍がないなら、召喚時効果持ちを優先（既にリスト順で考慮済み）
                return cmd
                
        # リストにないが召喚可能なものがあれば（念のため）
        fallback = self._find_command_by_type(commands, c.CommandType.SUMMON)
        if fallback: return fallback
        
        # セット (SET_MONST) のロジック
        # 守備的なモンスターや、リバース効果狙いならセット
        # atk.txt: "優先度8: モンスターのセット（攻撃力が低いもの）"
        set_cmds = [cmd for cmd in commands if cmd.command_type == c.CommandType.SET_MONST]
        if set_cmds:
            # 攻撃力が低い順（弱いものから壁にする）
            # コマンドからカード情報を引く必要があるが、ここでは簡易的にリストの先頭
            return set_cmds[0]
            
        return None

    def _logic_position_change(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """表示形式変更"""
        # 攻撃表示にする (TURN_ATK)
        turn_atk = [cmd for cmd in commands if cmd.command_type == c.CommandType.TURN_ATK]
        if turn_atk:
            # 攻撃力が高い順に攻撃表示に
            # （実装省略：ActionData生成時にカード詳細が必要）
            return turn_atk[0]
            
        # 守備表示にする (TURN_DEF)
        # 基本的にメイン1では攻撃したいので守備にはしないが、
        # 攻撃済みで守りたい場合（メイン2）などに使う。メイン1ではスルー。
        return None

    # =========================================================================
    # 選択・ターゲットロジック
    # =========================================================================
    
    def _select_card_target(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        カード選択要求時のロジック (atk.txtの優先度/確率を反映)
        """
        # ここでは「どのカードを選ぶか」のリストが commands に入っている
        # 状況（発動したカードの効果）に応じて判断を変える必要があるが、
        # `command_request` からは「どの効果の処理中か」が直接分からない場合がある。
        # 直前の行動ログや、現在チェーンブロックから推測する必要がある。
        
        # 汎用的な「強いカードを選ぶ」ロジック
        # 蘇生対象選択などを想定
        
        # 優先度付きIDリスト
        tier_s = [ID_BLUE_EYES, ID_PALADIN, ID_GENESIS_DRAGON]
        tier_a = [ID_ALEXANDRITE, ID_SAPPHIRE, ID_ASSAULT_WYVERN]
        tier_b = [ID_CAVE_DRAGON, ID_MASKED_DRAGON]
        
        # 1. Sティアを探す
        for cid in tier_s:
            cmd = self._find_command_by_card_id(commands, cid)
            if cmd: return cmd
            
        # 2. Aティア
        for cid in tier_a:
            cmd = self._find_command_by_card_id(commands, cid)
            if cmd: return cmd
            
        # 3. その他
        return commands[0]

    def _select_position(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """表示形式選択（召喚時など）"""
        # 基本的に攻撃表示 (1) を優先、壁なら守備 (2)
        # ID_ATTACK = 1, ID_DEFENSE = 4, ID_SET = 8 ? (ygoの実装依存)
        # ここではコマンドの文字情報などから推測するか、デフォルト攻撃
        return commands[0] # 仮

    # =========================================================================
    # ヘルパーメソッド
    # =========================================================================

    def _get_my_field(self, duel_state: DuelStateData) -> List[DuelCard]:
        return [c for c in duel_state.duel_card_table if c.player_id == 0] # 0: 自分

    def _get_opponent_field(self, duel_state: DuelStateData) -> List[DuelCard]:
        return [c for c in duel_state.duel_card_table if c.player_id == 1] # 1: 相手

    def _find_command_by_card_id(self, commands: List[CommandEntry], card_id: int, command_type: int = None) -> Optional[CommandEntry]:
        """指定されたCardIDを持つコマンドを探す"""
        for cmd in commands:
            if command_type is not None and cmd.command_type != command_type:
                continue
            # cmd.card_id が直接入っている場合と、table_index経由の場合がある
            # ここでは cmd.card_id を信頼する（ygo仕様）
            if cmd.card_id == card_id:
                return cmd
        return None

    def _find_command_by_type(self, commands: List[CommandEntry], command_type: int) -> Optional[CommandEntry]:
        for cmd in commands:
            if cmd.command_type == command_type:
                return cmd
        return None
    
    def _get_priority_command(self, commands: List[CommandEntry], priority_types: List[int]) -> CommandEntry:
        """優先順位リストに従ってコマンドを返す"""
        for p_type in priority_types:
            cmd = self._find_command_by_type(commands, p_type)
            if cmd: return cmd
        return commands[0]

    # --- フェイズメソッドのプレースホルダー（後半実装用） ---
    def _phase_battle(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        # 後半の実装で記述
        # 攻撃対象選択、ダメステの処理
        to_main2 = self._find_command_by_type(commands, c.CommandType.CHANGE_PHASE)
        if to_main2: return to_main2
        return commands[0]

    def _phase_main2(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        # 後半の実装で記述
        # 伏せカード最適化、エンド移行
        to_end = self._find_command_by_type(commands, c.CommandType.CHANGE_PHASE)
        if to_end: return to_end
        return commands[0]

    def _phase_end(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        # エンドフェイズ処理
        return commands[0]
    

    def _phase_battle(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """バトルフェイズ: 攻撃順序の最適化と攻撃宣言"""
        # 1. 攻撃コマンドの収集
        attack_cmds = [c for c in commands if c.command_type == c.CommandType.ATTACK]
        
        # 攻撃できるモンスターがいない場合、または攻撃終了したい場合
        if not attack_cmds:
            # バトル終了 -> メイン2へ
            to_main2 = self._find_command_by_type(commands, c.CommandType.CHANGE_PHASE)
            if to_main2: return to_main2
            return commands[0]

        duel_state = state.duel_state_data
        opp_field = self._get_opponent_field(duel_state)
        opp_monsters = [c for c in opp_field if c.position == c.PosId.MZONE]
        opp_faceup_monsters = [c for c in opp_monsters if c.face != c.Face.DOWN_DEFENSE]
        opp_set_cards = [c for c in opp_field if c.face == c.Face.DOWN] # 伏せカード
        
        my_field = self._get_my_field(duel_state)
        
        # 攻撃ロジック (atk.txt準拠)
        # 基本: 低ATK -> 高ATK (露払い -> 本命)
        
        # 攻撃可能な自分のモンスター情報を取得（コマンドから紐づけ）
        # ※簡易実装: コマンド順序をATKでソートする
        # 本来は command.table_index から DuelCard を引いて ATK を見る
        
        cmd_card_map = []
        for cmd in attack_cmds:
            # table_index は自分フィールドのインデックス
            if 0 <= cmd.table_index < len(duel_state.duel_card_table):
                card = duel_state.duel_card_table[cmd.table_index]
                cmd_card_map.append((cmd, card))
        
        # ソート基準の決定
        # 例外A: ワンショット圏内でバックが薄い -> 高ATKから (早期決着)
        # 例外B: 相手にボマー・ドラゴンがいる -> どうでもいいモンスターから当てる
        
        is_otk_range = False # 本来はダメージ計算が必要
        has_bomber = any(c.card_id == ID_BOMBER_DRAGON for c in opp_faceup_monsters)
        has_backrow = len([c for c in opp_field if c.position == c.PosId.SZONE]) > 0
        
        if not has_backrow and is_otk_range:
            # 高い順 (Reverse=True)
            cmd_card_map.sort(key=lambda x: x[1].atk_val, reverse=True)
        elif has_bomber:
            # キーフィニッシャー（最高打点）を温存するため、低い順、かつキー以外から
            # ここでは単純に低い順
            cmd_card_map.sort(key=lambda x: x[1].atk_val, reverse=False)
        else:
            # 基本: 低い順 (囮から)
            cmd_card_map.sort(key=lambda x: x[1].atk_val, reverse=False)

        # 最優先の攻撃コマンドを選択
        best_attack = cmd_card_map[0][0]
        
        # 攻撃対象選択 (SELECT_CARD) が発生する場合の処理は _select_card_target で行うが、
        # ここでATTACKコマンドを選ぶ時点で対象が決まっている場合もある（相手が1体のみ等）
        
        return best_attack

    def _phase_main2(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """メインフェイズ2: 次のターンへの布陣、伏せカード管理"""
        duel_state = state.duel_state_data
        
        # --- A. 召喚権の使い切り (サーチ/墓地肥やし) ---
        # バトルで盤面が変わった後、追加で出せるなら出す
        # atk.txt: "召喚権_残り: 儀式ピースの整備を最優先"
        summon_cmd = self._logic_normal_summon(state, commands)
        if summon_cmd:
            # メイン1と同じロジックだが、メイン2では「壁にする」意識も必要
            # ここでは共通ロジックを利用
            return summon_cmd

        # --- B. 白竜の聖騎士の効果 (メイン2での安全展開) ---
        # 攻撃後にリリースして青眼を出す
        paladin_cmd = self._logic_paladin_effect(state, commands)
        if paladin_cmd:
            return paladin_cmd

        # --- C. 蘇生罠・魔法のセット/発動 ---
        # atk.txt: "銀龍の轟咆はMP2で盤面を厚くしてエンド"
        # 速攻魔法の発動
        silvers_cry = self._find_command_by_card_id(commands, ID_SILVERS_CRY, command_type=c.CommandType.ACTIVATE)
        if silvers_cry:
            # 墓地に通常ドラゴンがいれば発動して壁/打点追加
            return silvers_cry

        # --- D. 魔法・罠のセット (伏せ過ぎないロジック) ---
        set_cmd = self._logic_set_cards_optimized(state, commands)
        if set_cmd:
            return set_cmd

        # --- E. 手札調整 (エンドフェイズへ) ---
        return self._get_priority_command(commands, [c.CommandType.CHANGE_PHASE, c.CommandType.PASS])

    def _logic_set_cards_optimized(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """
        atk.txt 付近の伏せ管理ロジック
        「全伏せ可」だが「割られ方まで設計」する。
        """
        set_cmds = [cmd for cmd in commands if cmd.command_type == c.CommandType.SET]
        if not set_cmds:
            return None
            
        duel_state = state.duel_state_data
        my_field = self._get_my_field(duel_state)
        my_backrow = [c for c in my_field if c.position == c.PosId.SZONE]
        
        # 既にセットされているカードIDを確認したいが、裏側なのでIDが見えない可能性がある。
        # 本来は `DuelLog` や自分の `hand` 履歴から推測するが、
        # ここでは「現在セットできる手札のカード」を評価する。
        
        # 優先してセットすべきカード
        priority_set_ids = [
            ID_TORRENTIAL_TRIBUTE, # 激流葬 (迎撃)
            ID_MIRROR_FORCE,       # ミラフォ (迎撃)
            ID_RETURN_TO_FRONT,    # 戦線復帰 (蘇生罠)
            ID_CALL_OF_HAUNTED,    # リビデ (蘇生罠)
            ID_BOOK_OF_MOON,       # 月の書 (防御/妨害)
            ID_LANCE,              # 聖槍 (防御)
            ID_DUST_TORNADO,       # 砂塵 (囮/妨害)
            ID_CYCLONE,            # サイクロン (囮/妨害)
            # 銀龍の轟咆 (QP) は相手ターンに使いたいならセット
            ID_SILVERS_CRY,
            ID_SHRINK              # 収縮
        ]
        
        # 大嵐 (Heavy Storm) は原則セットしない (atk.txt: "原則温存")
        # ただし、手札抹殺ケアなどで伏せる場合もあるが、ここではロジック通り温存。
        
        # セット候補のコマンドをフィルタリング
        # コマンドに対応するカードIDを取得して判定
        
        best_set_cmd = None
        
        for pid in priority_set_ids:
            # 既に同名カードがセットされているかは、厳密には分からないが
            # 「まだセットしていない」ものを優先したい。
            # ここでは単純に優先度リスト順にセットする。
            
            cmd = self._find_command_by_card_id(commands, pid, command_type=c.CommandType.SET)
            if cmd:
                # 伏せすぎ抑制ロジック
                # もし「次ターン大嵐プラン」があるなら、伏せを2枚程度に抑える等の調整
                # atk.txt: "戦術_全伏せ許容" ならガンガン伏せる
                
                # ここでは安全策として、バックが4枚以下なら伏せる (ルール上5枚まで)
                if len(my_backrow) < 5:
                    return cmd
        
        return None

    def _phase_end(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """エンドフェイズ: 手札調整とフリーチェーン"""
        duel_state = state.duel_state_data
        
        # 1. 優先して処理すべきフリーチェーン発動
        # atk.txt source 48: "次ターンの打点/素材を先に用意しておきたい" -> 銀龍の轟咆
        silvers_cry = self._find_command_by_card_id(commands, ID_SILVERS_CRY, command_type=c.CommandType.ACTIVATE)
        if silvers_cry:
            # 墓地有効なら発動
            return silvers_cry
            
        # 2. 手札調整 (ディスカード)
        # selection_type が DECIDE や SELECT_CARD になる可能性があるが、
        # ここではコマンド選択としての処理
        # もし「捨てるカードを選べ」というSelectionTypeなら、_select_card_target で処理される。
        
        # ここでのコマンドは「パス」してターン終了するのが基本
        return self._get_priority_command(commands, [c.CommandType.PASS])

    # =========================================================================
    # ディスカードロジック (手札調整時のカード選択)
    # =========================================================================

    def _logic_discard_priority(self, commands: List[CommandEntry]) -> CommandEntry:
        """
        手札コストやエンドフェイズの手札調整で捨てるべきカードを選択
        atk.txt 
        1. 通常ドラゴン (銀龍の弾)
        2. 大型ドラゴン (蘇生札の弾)
        3. 過剰な儀式ピース
        4. 腐っているカード
        """
        # 捨てて良い順リスト
        discard_priority = [
            ID_BLUE_EYES,      # 最優先：墓地へ送って蘇生対象に
            ID_ALEXANDRITE,
            ID_SAPPHIRE,
            ID_CAVE_DRAGON,
            ID_KIDMODO,        # コドモドラゴン（効果発動狙い）
            ID_GENESIS_DRAGON, # 高レベル
            ID_ADVANCED_RITUAL_ART, # 儀式魔法（余っていれば）
            ID_WHITE_DRAGON_RITUAL,
        ]
        
        # 重要なカード（捨てたくない）
        keep_ids = [
            ID_MONSTER_REBORN,
            ID_HARPIE,
            ID_CYCLONE,
            ID_MIRROR_FORCE,
            ID_TORRENTIAL_TRIBUTE
        ]

        # 優先度リストにあるものを探す
        for card_id in discard_priority:
            cmd = self._find_command_by_card_id(commands, card_id)
            if cmd: return cmd
            
        # リストになく、かつKeepリストにもないもの（中間のカード）
        for cmd in commands:
            if cmd.card_id not in keep_ids:
                return cmd
                
        # どうしようもない場合は先頭
        return commands[0]

    # _select_card_target をオーバーライドしてディスカードロジックを組み込む
    # （前のコードのメソッドを拡張）
    def _select_card_target_extended(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        カード選択の拡張版
        - 蘇生対象選択 -> 強いモンスター
        - 手札コスト選択 -> 墓地で得するモンスター
        - 攻撃対象選択 -> _phase_battleのロジックに従う
        """
        # コンテキスト判断が難しいが、コマンドの card_location などで判定
        # 手札からの選択かつ、自分フィールド/墓地への移動でないならコストの可能性が高い
        
        # 簡易判定: 墓地のモンスターを選ぶなら蘇生
        first_cmd = commands[0]
        # table_index等から場所を特定するロジックが必要だが、ここでは card_id ベースのヒューリスティック
        
        # 蘇生対象の優先度 (Sティア)
        revive_priority = [ID_BLUE_EYES, ID_PALADIN, ID_GENESIS_DRAGON, ID_ALEXANDRITE]
        for card_id in revive_priority:
            cmd = self._find_command_by_card_id(commands, card_id)
            if cmd: return cmd
            
        return commands[0]