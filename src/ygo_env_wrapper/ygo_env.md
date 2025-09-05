# YgoEnv
## note

### reset()は実装していない
`UdiIO()`のコンストラクタで確か、episodeを設定できるので、指定したepisode分初期化も`UdiIO`側で行ってくれるため (違う場合は教えてください)

### step()の仕様
* `is_command_required()`がFalseでも次状態などを返す
    - Chainの仕様などを考えると、チェーン発動中でも自分が行動をしない場合もあるが、チェーンで起こったイベントは認知したい

    - つまり、main()では、is_command_requireid()がTrueの時のみ行動選択をするように実装する
* 行動データ`ActionData`の定義
    - 実際に行動を必要としないが、状態が常に変わり続けることを考慮する
    - 