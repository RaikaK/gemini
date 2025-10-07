# LLM Agent　Desing
LLMエージェントの設計

## 各フェーズごとにプロンプトを帰る
用意するプロンプトメソッド
```python
class PromptGenerator:
    def on_standby_phase() # スタンバイフェーズでの意思決定プロンプト
    def on_first_main_phaset() # メインフェイズ1での意思決定プロンプト
    def on_battle_phase() # バトルフェイズでの意思決定プロンプト
    def on_second_main_phase() # メインフェーズ2での意思決定プロンプト
    def on_chain_process() # チェイン時の意思決定プロンプト
    def on_end_phase() # ターン終了時の意思決定プロンプト
```

LLMによる意思決定は、行動の選択肢が2つ以上の時のみ推論させる。

実際に選択した行動はプロンプトに組み込みたいが、
おそらく、command_request.command_entry_logで過去の行動をテキストに変換して、プロンプトに組み込む方針になりそう
