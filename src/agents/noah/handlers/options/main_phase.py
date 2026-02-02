"""メインフェイズ (SelectionType:1) & 値無し (SelectionId:-1)"""

OPTIONS = {
    "card_id:None|command_type:CHANGE_PHASE|phase:BATTLE": {
        "card_id": -1,
        "command_type": 3,
        "phase": 3,
    },
    "card_id:None|command_type:CHANGE_PHASE|phase:END": {
        "card_id": -1,
        "command_type": 3,
        "phase": 5,
    },
    "card_id:洞窟に潜む竜|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1001,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:洞窟に潜む竜|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1001,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:洞窟に潜む竜|command_type:TURN_ATK|phase:NO_VALUE": {
        "card_id": 1001,
        "command_type": 14,
        "phase": -1,
    },
    "card_id:洞窟に潜む竜|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1001,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:サファイアドラゴン|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1002,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:サファイアドラゴン|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1002,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:サファイアドラゴン|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1002,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:アレキサンドライドラゴン|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1003,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:アレキサンドライドラゴン|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1003,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:アレキサンドライドラゴン|command_type:TURN_ATK|phase:NO_VALUE": {
        "card_id": 1003,
        "command_type": 14,
        "phase": -1,
    },
    "card_id:アレキサンドライドラゴン|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1003,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:青眼の白龍|command_type:REVERSE|phase:NO_VALUE": {
        "card_id": 1004,
        "command_type": 9,
        "phase": -1,
    },
    "card_id:青眼の白龍|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1004,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:青眼の白龍|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1004,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:青眼の白龍|command_type:TURN_ATK|phase:NO_VALUE": {
        "card_id": 1004,
        "command_type": 14,
        "phase": -1,
    },
    "card_id:青眼の白龍|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1004,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:アサルトワイバーン|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1005,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:アサルトワイバーン|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1005,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:アサルトワイバーン|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1005,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:強欲な壺|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1006,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:強欲な壺|command_type:SET|phase:NO_VALUE": {
        "card_id": 1006,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:大嵐|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1007,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:大嵐|command_type:SET|phase:NO_VALUE": {
        "card_id": 1007,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:ライトニング・ボルテックス|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1008,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:ライトニング・ボルテックス|command_type:SET|phase:NO_VALUE": {
        "card_id": 1008,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:早すぎた埋葬|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1009,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:早すぎた埋葬|command_type:SET|phase:NO_VALUE": {
        "card_id": 1009,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:サイクロン|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1010,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:サイクロン|command_type:SET|phase:NO_VALUE": {
        "card_id": 1010,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:収縮|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1011,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:収縮|command_type:SET|phase:NO_VALUE": {
        "card_id": 1011,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:銀龍の轟咆|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1012,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:銀龍の轟咆|command_type:SET|phase:NO_VALUE": {
        "card_id": 1012,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:聖なるバリア －ミラーフォース－|command_type:SET|phase:NO_VALUE": {
        "card_id": 1013,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:砂塵の大竜巻|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1014,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:砂塵の大竜巻|command_type:SET|phase:NO_VALUE": {
        "card_id": 1014,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:激流葬|command_type:SET|phase:NO_VALUE": {
        "card_id": 1015,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:強化蘇生|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1016,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:強化蘇生|command_type:SET|phase:NO_VALUE": {
        "card_id": 1016,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:仮面竜|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1017,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:仮面竜|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1017,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:仮面竜|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1017,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:ボマー・ドラゴン|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1018,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:ボマー・ドラゴン|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1018,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:ボマー・ドラゴン|command_type:TURN_ATK|phase:NO_VALUE": {
        "card_id": 1018,
        "command_type": 14,
        "phase": -1,
    },
    "card_id:ボマー・ドラゴン|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1018,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:コドモドラゴン|command_type:REVERSE|phase:NO_VALUE": {
        "card_id": 1019,
        "command_type": 9,
        "phase": -1,
    },
    "card_id:コドモドラゴン|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1019,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:コドモドラゴン|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1019,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:コドモドラゴン|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1019,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:センジュ・ゴッド|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1020,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:センジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1020,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:センジュ・ゴッド|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1020,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:ソニックバード|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1021,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:ソニックバード|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1021,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:ソニックバード|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1021,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:マンジュ・ゴッド|command_type:REVERSE|phase:NO_VALUE": {
        "card_id": 1022,
        "command_type": 9,
        "phase": -1,
    },
    "card_id:マンジュ・ゴッド|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1022,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:マンジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1022,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:マンジュ・ゴッド|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1022,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:創世の竜騎士|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1023,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:創世の竜騎士|command_type:SET_MONST|phase:NO_VALUE": {
        "card_id": 1023,
        "command_type": 11,
        "phase": -1,
    },
    "card_id:創世の竜騎士|command_type:SUMMON|phase:NO_VALUE": {
        "card_id": 1023,
        "command_type": 12,
        "phase": -1,
    },
    "card_id:創世の竜騎士|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1023,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:白竜の聖騎士|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1024,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:白竜の聖騎士|command_type:TURN_ATK|phase:NO_VALUE": {
        "card_id": 1024,
        "command_type": 14,
        "phase": -1,
    },
    "card_id:白竜の聖騎士|command_type:TURN_DEF|phase:NO_VALUE": {
        "card_id": 1024,
        "command_type": 15,
        "phase": -1,
    },
    "card_id:死者蘇生|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1025,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:死者蘇生|command_type:SET|phase:NO_VALUE": {
        "card_id": 1025,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:白竜降臨|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1026,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:白竜降臨|command_type:SET|phase:NO_VALUE": {
        "card_id": 1026,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:高等儀式術|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1027,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:高等儀式術|command_type:SET|phase:NO_VALUE": {
        "card_id": 1027,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:月の書|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1028,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:月の書|command_type:SET|phase:NO_VALUE": {
        "card_id": 1028,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:禁じられた聖槍|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1029,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:禁じられた聖槍|command_type:SET|phase:NO_VALUE": {
        "card_id": 1029,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:戦線復帰|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1030,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:戦線復帰|command_type:SET|phase:NO_VALUE": {
        "card_id": 1030,
        "command_type": 10,
        "phase": -1,
    },
    "card_id:リビングデッドの呼び声|command_type:ACTIVATE|phase:NO_VALUE": {
        "card_id": 1031,
        "command_type": 0,
        "phase": -1,
    },
    "card_id:リビングデッドの呼び声|command_type:SET|phase:NO_VALUE": {
        "card_id": 1031,
        "command_type": 10,
        "phase": -1,
    },
}
