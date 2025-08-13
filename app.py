# uv pip install google-generativeai
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import random
import re
import json
import gc
import sys
import os
import time
import google.generativeai as genai
from typing import Optional, Dict, List, Any
import google.generativeai as genai # gemma用

app = Flask(__name__)
CORS(app)

# ログ設定 - より詳細なログ出力
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gemini API設定
GEMINI_API_KEY = "AIzaSyA_XleL8lGvzJAE1QTpfS429amLos6jqgc"

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. Please set it for faster generation.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")

# LLamaモデル設定（面接応答生成用のフォールバック）
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
model = None
tokenizer = None

# デバッグ情報を保存するグローバル変数
debug_logs = []

def add_debug_log(log_type, content):
    """デバッグログを追加"""
    global debug_logs
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    debug_logs.append({
        'timestamp': timestamp,
        'type': log_type,
        'content': content
    })
    # 最新の10件のみ保持
    debug_logs = debug_logs[-10:]
    logger.debug(f"[{log_type}] {content}")

def cleanup_gpu_memory():
    """GPU メモリをクリーンアップ"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def initialize_model():
    """モデルの初期化（面接応答用のみ）"""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        logger.info("Model already loaded, skipping initialization")
        return True
    
    try:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. GPU is required.")
            sys.exit(1)
        
        cleanup_gpu_memory()
        
        device = torch.device("cuda")
        logger.info(f"Starting LLama model loading: {MODEL_NAME}")
        logger.info(f"Using device: {device}")
        
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1e9
        current_memory = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU Memory: {total_memory:.1f} GB (Current: {current_memory:.1f} GB)")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
        )
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            max_memory={
                0: "12GB",
                "cpu": "20GB"
            },
            low_cpu_mem_usage=True,
        )
        
        logger.info("Model loaded successfully")
        logger.info(f"Model info: {model.config.name_or_path}")
        logger.info(f"Quantization: 8bit (BitsAndBytesConfig)")
        logger.info(f"Data type: {model.dtype}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"GPU Memory usage: {memory_allocated:.1f} GB / {memory_reserved:.1f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load LLama model: {str(e)}", exc_info=True)
        cleanup_gpu_memory()
        logger.error("Fallback is disabled. LLama model is required.")
        sys.exit(1)

class GeminiGenerator:
    """Gemini APIを使用したデータ生成クラス"""
    
    @staticmethod
    def is_available() -> bool:
        """Gemini APIが利用可能かチェック"""
        return GEMINI_API_KEY is not None
    
    @staticmethod
    def generate_with_gemini(prompt: str, max_retries: int = 3) -> Optional[str]:
        """Gemini APIを使用してテキスト生成"""
        if not GeminiGenerator.is_available():
            return None
            
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-06-17')
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]
                
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if response.text:
                    logger.debug(f"Gemini generated response: {response.text[:100]}...")
                    return response.text.strip()
                else:
                    logger.warning(f"Gemini response was blocked or empty. Attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Gemini generation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    logger.error(f"All Gemini generation attempts failed after {max_retries} retries")
                    return None
                
                time.sleep(1)
        
        return None
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
        """レスポンスからJSONを抽出"""
        match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from markdown block: {e}")

        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from outer brackets: {e}")

        logger.warning("Could not find a valid JSON object in the response.")
        return None
    
    @staticmethod
    def generate_companies(count: int = 1) -> List[Dict[str, str]]:
        """企業情報を生成（Gemini API使用）"""
        logger.info(f"Generating {count} company information with Gemini API...")
        
        if not GeminiGenerator.is_available():
            logger.warning("Gemini API not available, using fallback data")
            return GeminiGenerator._get_fallback_companies(count)
        
        prompt = f"""
日本の架空の企業{count}社の情報を生成してください。

以下のJSON形式で出力してください：

```json
{{
  "companies": [
    {{
      "name": "企業名",
      "business": "事業内容（50字程度）",
      "revenue": "年商XX億円",
      "employees": "従業員数XX名",
      "founded": "設立XXXX年",
      "location": "本社：XX都XX区",
      "vision": "企業ビジョン（30字程度）",
      "products": "主力製品・サービス（30字程度）",
      "culture": "企業文化（30字程度）",
      "recent_news": "最近のニュース・動向（30字程度）",
      "competitive_advantage": "競合優位性（30字程度）",
      "ceo_message": "CEO・代表メッセージ（30字程度）",
      "expansion_plan": "事業展開計画（30字程度）",
      "awards": "受賞歴・評価（30字程度）",
      "partnerships": "パートナーシップ・提携（30字程度）"
    }}
  ]
}}
```

要件：
- 実在しない架空の企業名を使用
- 多様な業界（IT、製造業、サービス業など）
- リアルな規模感（従業員数100-5000名程度）
- 現代的な日本企業として設定
"""
        
        response = GeminiGenerator.generate_with_gemini(prompt)
        
        if response:
            data = GeminiGenerator.extract_json_from_response(response)
            
            if data and 'companies' in data and isinstance(data['companies'], list) and len(data['companies']) >= count:
                companies = data['companies'][:count]
                logger.info(f"Successfully generated {len(companies)} companies with Gemini")
                for i, company in enumerate(companies):
                    logger.info(f"  {i+1}. {company.get('name', 'Unknown')}")
                return companies
        
        logger.warning("Gemini company generation failed, using fallback data")
        return GeminiGenerator._get_fallback_companies(count)
    
    @staticmethod
    def generate_candidates(count: int = 3) -> List[Dict[str, str]]:
        """就活生情報を生成（Gemini API使用）"""
        logger.info(f"Generating {count} candidate information with Gemini API...")
        
        if not GeminiGenerator.is_available():
            logger.warning("Gemini API not available, using fallback data")
            return GeminiGenerator._get_fallback_candidates(count)
        
        prompt = f"""
日本の架空の就活生{count}人の情報を生成してください。

以下のJSON形式で出力してください：

```json
{{
  "candidates": [
    {{
      "name": "日本人の氏名",
      "university": "大学名学部名",
      "gakuchika": "学生時代に力を入れたこと（100字程度の具体的なエピソード）",
      "interest": "興味のある分野・職種",
      "strength": "強み・能力",
      "preparation": "high",
      "mbti": "MBTIタイプ（例：INTJ、ESFJなど）",
    }}
  ]
}}
```

要件：
- 多様な大学・学部（国公立・私立・地方大学含む）
- 具体的で説得力のあるガクチカエピソード
- 異なる強みや興味分野を持つ学生
- リアルな日本の就活生として設定
- preparationは全て"high"に設定（後で調整します）
"""
        
        response = GeminiGenerator.generate_with_gemini(prompt)
        
        if response:
            data = GeminiGenerator.extract_json_from_response(response)
            
            if data and 'candidates' in data and isinstance(data['candidates'], list) and len(data['candidates']) >= count:
                candidates = data['candidates'][:count]
                for candidate in candidates:
                    candidate['preparation'] = "high"
                    
                logger.info(f"Successfully generated {len(candidates)} candidates with Gemini")
                for i, candidate in enumerate(candidates):
                    logger.info(f"  {i+1}. {candidate.get('name', 'Unknown')}")
                return candidates
        
        logger.warning("Gemini candidate generation failed, using fallback data")
        return GeminiGenerator._get_fallback_candidates(count)
    
    @staticmethod
    def _get_fallback_companies(count: int) -> List[Dict[str, str]]:
        """フォールバック用企業データ（詳細情報付き）"""
        fallback_companies = [
            {
                "name": "テックイノベーション株式会社",
                "business": "AI・機械学習を活用したSaaS開発と企業向けDXソリューション",
                "revenue": "年商50億円",
                "employees": "従業員数300名",
                "founded": "設立2015年",
                "location": "本社：東京都渋谷区",
                "vision": "AIで世界をもっと便利に",
                "products": "AIアシスタント「SmartHelper」",
                "culture": "フラットな組織文化、リモートワーク推進",
                "recent_news": "新製品リリースで売上30%増",
                "competitive_advantage": "独自のAI技術と高い顧客満足度",
                "ceo_message": "技術で社会課題を解決する",
                "expansion_plan": "2025年に海外展開予定",
                "awards": "AI技術賞2024受賞",
                "partnerships": "大手IT企業と戦略的提携"
            },
            {
                "name": "グリーンエナジーソリューションズ",
                "business": "再生可能エネルギーシステムの開発・販売・保守",
                "revenue": "年商120億円",
                "employees": "従業員数800名",
                "founded": "設立2010年",
                "location": "本社：大阪府大阪市",
                "vision": "持続可能な未来をエネルギーで実現",
                "products": "太陽光発電システム、蓄電池",
                "culture": "環境意識の高いチーム、社会貢献重視",
                "recent_news": "海外展開を本格化、アジア市場に参入",
                "competitive_advantage": "技術力と環境への取り組み",
                "ceo_message": "地球環境保護に貢献する企業",
                "expansion_plan": "アジア太平洋地域への事業拡大",
                "awards": "環境経営大賞2024受賞",
                "partnerships": "国内外の再エネ企業と連携"
            }
        ]
        return fallback_companies[:count] if count <= len(fallback_companies) else fallback_companies * ((count // len(fallback_companies)) + 1)
    
    @staticmethod
    def _get_fallback_candidates(count: int) -> List[Dict[str, str]]:
        """フォールバック用候補者データ"""
        fallback_candidates = [
            {
                "name": "田中太郎",
                "university": "東京大学経済学部",
                "gakuchika": "学生団体でのリーダー経験。100人規模のイベントを企画・運営し、参加者満足度95%を達成。チームメンバーとの調整や予算管理に苦労したが、粘り強く取り組んだ。",
                "interest": "事業戦略立案・経営企画",
                "strength": "リーダーシップと分析力",
                "preparation": "high"
            },
            {
                "name": "佐藤花子",
                "university": "慶應義塾大学商学部",
                "gakuchika": "長期インターンで営業成績トップを達成。新規顧客開拓で月間目標を150%達成し、部署全体のモチベーション向上にも貢献した。顧客の課題を深く理解することの重要性を学んだ。",
                "interest": "マーケティング・営業企画",
                "strength": "コミュニケーション能力と課題解決力",
                "preparation": "high"
            },
            {
                "name": "鈴木次郎",
                "university": "早稲田大学理工学部",
                "gakuchika": "プログラミングコンテストで全国3位入賞。チーム開発でリーダーを務め、効率的なアルゴリズム設計と実装を担当した。技術的な議論を通じてチームワークの大切さを実感した。",
                "interest": "技術開発・エンジニアリング",
                "strength": "技術力と問題解決能力",
                "preparation": "high"
            }
        ]
        return fallback_candidates[:count] if count <= len(fallback_candidates) else fallback_candidates * ((count // len(fallback_candidates)) + 1)

class LLamaGenerator:
    """LLamaモデル生成クラス（改良版ストリーミング対応）"""
    
    @staticmethod
    def generate_with_llama_stream(prompt, max_tokens=256, temperature=0.8, max_retries=3):
        """ストリーミング対応のLLama生成（文字化け対策版）"""
        for attempt in range(max_retries):
            try:
                device = torch.device("cuda")
                cleanup_gpu_memory()
                
                system_prompt = "あなたは優秀な日本語AIアシスタントです。指示に従って適切に回答してください。"
                full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                add_debug_log("LLAMA_STREAM_INPUT", f"Prompt length: {len(full_prompt)} chars\nFull prompt:\n{full_prompt}")
                
                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                logger.debug(f"Input tokens: {inputs['input_ids'].shape[1]}")
                
                with torch.no_grad():
                    generated_ids = inputs['input_ids'].clone()
                    past_key_values = None
                    accumulated_tokens = []
                    complete_response = ""
                    
                    for step in range(max_tokens):
                        if step == 0:
                            outputs = model(
                                input_ids=generated_ids,
                                use_cache=True,
                                return_dict=True
                            )
                        else:
                            outputs = model(
                                input_ids=generated_ids[:, -1:],
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True
                            )
                        
                        past_key_values = outputs.past_key_values
                        logits = outputs.logits[0, -1, :]
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        
                        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                        accumulated_tokens.append(next_token.item())
                        
                        try:
                            partial_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=True)
                            
                            if partial_text and len(partial_text) > len(complete_response):
                                new_chunk = partial_text[len(complete_response):]
                                
                                if new_chunk and not new_chunk.endswith('�') and new_chunk.isprintable():
                                    complete_response = partial_text
                                    yield new_chunk
                                
                        except Exception as decode_error:
                            logger.debug(f"Decode error at step {step}: {decode_error}")
                            continue
                
                final_response = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                add_debug_log("LLAMA_STREAM_OUTPUT", f"Final response length: {len(final_response)} chars\nFinal response:\n{final_response}")
                
                yield ("__COMPLETE__", final_response)
                
                del inputs, generated_ids
                cleanup_gpu_memory()
                return
                
            except Exception as e:
                logger.warning(f"Streaming generation attempt {attempt + 1} failed: {str(e)}")
                add_debug_log("LLAMA_STREAM_ERROR", f"Attempt {attempt + 1} failed: {str(e)}")
                cleanup_gpu_memory()
                
                if attempt == max_retries - 1:
                    logger.error(f"All streaming generation attempts failed after {max_retries} retries")
                    raise e
                
                time.sleep(1)

class CompanyKnowledgeManager:
    """企業情報のランダムフィルタリング制御クラス"""

    def _punch_holes_in_string(text: str, percentage: float, placeholder: str = '_') -> str:
        """
        文字列を単語単位で穴抜きする。
        指定された割合の単語を '_' の連続に置換する。
        「」内の単語に対しては、穴抜き率を低減する。
        非単語文字（スペース、句読点など）は穴抜きしない。
        """
        if not isinstance(text, str) or not text:
            return text

        result_tokens = []
        
        # テキストを「」で囲まれた部分とそれ以外の部分に分割
        segments = re.split(r'(「.*?」)', text)
        
        for segment in segments:
            if segment.startswith('「') and segment.endswith('」') and len(segment) > 1:
                # 括弧内の部分（例: 「SmartHelper」）
                inner_text = segment[1:-1]
                # 括弧内の穴抜き率を低減 (通常の1/3に)
                effective_percentage = percentage / 3.0 if percentage > 0 else 0.0
                
                # 括弧内の単語と非単語文字を識別
                # 日本語単語、英数字単語、その他の非空白文字（句読点など）
                tokens = re.findall(r'[一-龯ぁ-んァ-ヶ々]+|[a-zA-Z0-9]+|\S', inner_text)
                
                punched_inner_tokens = []
                for token in tokens:
                    # トークンが単語と見なせるもので、かつランダムで穴抜き対象になった場合
                    # （ここでは日本語文字または英数字の連続を単語と定義）
                    if (re.match(r'[一-龯ぁ-んァ-ヶ々]+|[a-zA-Z0-9]+', token) and
                        random.random() < effective_percentage):
                        punched_inner_tokens.append('_' * len(token)) # ここを修正
                    else:
                        punched_inner_tokens.append(token)
                
                result_tokens.append('「' + "".join(punched_inner_tokens) + '」')
            else:
                # 括弧外の通常のテキスト部分
                # 日本語単語、英数字単語、その他の非空白文字（句読点など）
                tokens = re.findall(r'[一-龯ぁ-んァ-ヶ々]+|[a-zA-Z0-9]+|\S', segment)
                
                punched_regular_tokens = []
                for token in tokens:
                    # トークンが単語と見なせるもので、かつランダムで穴抜き対象になった場合
                    if (re.match(r'[一-龯ぁ-んァ-ヶ々]+|[a-zA-Z0-9]+', token) and
                        random.random() < percentage):
                        punched_regular_tokens.append('_' * len(token)) # ここを修正
                    else:
                        punched_regular_tokens.append(token)
                result_tokens.append("".join(punched_regular_tokens))
        
        return "".join(result_tokens)

    
    @staticmethod
    def get_company_info_for_candidate(company, preparation_level, candidate_name):
        """
        企業情報をランダムにフィルタリングして学生に渡す
        - high: 100%の完全情報
        - high-middle: 85%の情報（ランダムに15%除外）
        - middle: 70%の情報（ランダムに30%除外）
        """
        
        # 候補者名を元にしたシード設定（一貫性のため）
        random.seed(hash(candidate_name + str(preparation_level)) % 1000000)
        
        # 全企業情報のフィールドリスト
        all_fields = [
            "name", "business", "revenue", "employees", "founded", "location",
            "vision", "products", "culture", "recent_news", "competitive_advantage",
            "ceo_message", "expansion_plan", "awards", "partnerships"
        ]
        
        # 必須フィールド（必ず含める）
        essential_fields = ["name", "business", "location"]
        
        # フィルタリング可能フィールド
        filterable_fields = [f for f in all_fields if f not in essential_fields]
        
        if preparation_level == "high":  # 100%の情報
            filtered_info = {k: v for k, v in company.items() if k in all_fields and v}
            filtered_info.update({
                "knowledge_coverage": "完璧な企業研究（100%）",
                "info_accuracy": "100%正確",
                "detail_level": "最新動向・詳細情報まで完全把握"
            })
            logger.info(f"{candidate_name} (high): 全フィールド {len(filtered_info)} 項目提供")

        elif preparation_level == "high-middle":  # 85%の情報（フィルタリング可能フィールドの文字列15%を穴抜き）
                punched_info = {}
                punch_percentage = 0.15 # 15%の文字を穴抜きする割合

                for key, value in company.items():
                    if key in essential_fields:
                        punched_info[key] = value
                    elif key in filterable_fields:
                        # フィルタリング可能フィールドかつ文字列の場合のみ穴抜きを適用
                        if isinstance(value, str):
                            punched_info[key] = CompanyKnowledgeManager._punch_holes_in_string(value, punch_percentage)
                        else:
                            # フィルタリング可能フィールドだが文字列でない場合はそのまま保持
                            punched_info[key] = value
                    else:
                        # all_fieldsに含まれない予期せぬキーはそのまま保持（通常は発生しない想定）
                        punched_info[key] = value
                
                filtered_info = punched_info # 穴抜きされた情報を設定
                filtered_info.update({
                    "knowledge_coverage": "高レベル企業研究（85%）",
                    "info_accuracy": "85%正確（一部文字が不明瞭）", # 説明文を調整
                    "detail_level": "詳細情報の大部分を把握（一部文字が不明瞭）" # 説明文を調整
                })
                logger.info(f"{candidate_name} (high-middle): フィルタリング可能フィールドに {punch_percentage*100}% の文字穴抜きを適用")
            
        elif preparation_level == "middle":  # 70%の情報
                punched_info = {}
                punch_percentage = 0.30 # 30%の文字を穴抜きする割合

                for key, value in company.items():
                    if key in essential_fields:
                        punched_info[key] = value
                    elif key in filterable_fields:
                        # フィルタリング可能フィールドかつ文字列の場合のみ穴抜きを適用
                        if isinstance(value, str):
                            punched_info[key] = CompanyKnowledgeManager._punch_holes_in_string(value, punch_percentage)
                        else:
                            # フィルタリング可能フィールドだが文字列でない場合はそのまま保持
                            punched_info[key] = value
                    else:
                        # all_fieldsに含まれない予期せぬキーはそのまま保持（通常は発生しない想定）
                        punched_info[key] = value
                
                filtered_info = punched_info # 穴抜きされた情報を設定
                filtered_info.update({
                    "knowledge_coverage": "中レベル企業研究（70%）",
                    "info_accuracy": "70%正確",
                    "detail_level": "基本情報と一部詳細を把握"
                })
                logger.info(f"{candidate_name} (middle): フィルタリング可能フィールドに {punch_percentage*100}% の文字穴抜きを適用")
        
        # ランダムシードをリセット
        random.seed()
        
        return filtered_info
    
class InstructionPromptManager:
    """回答方針作成クラス"""
    @staticmethod
    def create_instruction_prompt(preparation_level):
        if preparation_level == "high":
            return """
- 非常に高い志望度と熱意を必ず示してください。
- 他の就活生に負けない強い意欲を表現してください。
- 知っている具体的な情報は積極的に言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを強く表現してください。"""
        elif preparation_level == "high-middle":
            return """
- 高い志望度と熱意を必ず示してください。
- 他の就活生に負けない意欲を表現してください。
- 知っている具体的な情報は積極的に言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを表現してください。

- 企業情報の不足している部分(_)は推測しながら話してください。

"""
        else : 
            return  """
- そこそこ高い志望度と熱意を必ず示してください。
- 他の就活生に負けない意欲をなるべく表現してください。
- 知っている具体的な情報は言及してください。
- 知らない情報については前向きな推測や業界一般論で補ってください。
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください。
- この企業で成長したい気持ちを表現してください。

- 企業情報の不足している部分(_)は推測しながら話してください。

"""

class LLamaInterviewResponseGenerator:
    """High/High-Middle/Middleでの応答生成"""
    
    @staticmethod
    def generate_interview_response_stream(question, candidate, company, conversation_history, all_conversations=None):
        """全員が志望度高く見せるが僅かな情報差が出る応答生成"""
        logger.info(f"Generating subtle difference response for {candidate['name']} (prep: {candidate['preparation']})")
        
        company_info = CompanyKnowledgeManager.get_company_info_for_candidate(
            company, candidate["preparation"], candidate["name"]
        )
        
        enhanced_history = LLamaInterviewResponseGenerator._build_enhanced_conversation_history(
            candidate, conversation_history, all_conversations
        )
        
        # プロンプト：全員が高い志望度で僅かな情報差のみ
        prompt = f"""
あなたは {candidate["name"]} という日本の就活生です。この企業に絶対に入社したく、面接官に志望度の高さを強くアピールしたいと考えています。
企業研究を熱心に行いましたが、情報収集には限界があります。知っている情報は具体的に、知らない情報は前向きな推測や一般論で補って回答してください。

# あなたのプロフィール
- 氏名: {candidate["name"]}
- 大学: {candidate["university"]}
- 強み: {candidate["strength"]}
- ガクチカ: {candidate["gakuchika"]}
- MBTI: {candidate["mbti"]}

# あなたが調べて得た企業情報（{company_info.get("knowledge_coverage")}）
{LLamaInterviewResponseGenerator._format_available_company_info(company_info)}

# 回答の重要な方針
{InstructionPromptManager.create_instruction_prompt(candidate["preparation"])}

# これまでの会話
{enhanced_history}

# 面接官からの質問
{question}

---
{candidate["name"]} として、最高レベルの志望度と熱意を示しながら、150文字程度で自然な日本語で回答してください。
この企業への強い憧れと、絶対に入社したい気持ちを表現してください。
回答のみを出力し、説明や前置きは不要です。
"""
        
        # ストリーミング生成
        complete_response = ""
        for chunk_or_complete in LLamaGenerator.generate_with_llama_stream(prompt, max_tokens=256, temperature=0.75):
            if isinstance(chunk_or_complete, tuple) and chunk_or_complete[0] == "__COMPLETE__":
                complete_response = chunk_or_complete[1]
                break
            else:
                yield chunk_or_complete
        
        if complete_response:
            yield ("__COMPLETE__", complete_response)
    
    @staticmethod
    def _format_available_company_info(company_info):
        """利用可能な企業情報のフォーマット"""
        info_lines = []
        
        # 利用可能な情報のみを表示
        field_labels = {
            "name": "企業名",
            "business": "事業内容", 
            "revenue": "売上高",
            "employees": "従業員数",
            "founded": "設立年",
            "location": "本社所在地",
            "vision": "企業ビジョン",
            "products": "主力製品・サービス",
            "culture": "企業文化",
            "recent_news": "最近のニュース",
            "competitive_advantage": "競合優位性",
            "ceo_message": "CEO・代表メッセージ",
            "expansion_plan": "事業展開計画",
            "awards": "受賞歴・評価",
            "partnerships": "パートナーシップ・提携"
        }
        
        for field, label in field_labels.items():
            if company_info.get(field):
                info_lines.append(f"- {label}: {company_info[field]}")
        
        return "\n".join(info_lines) if info_lines else "- 企業名と基本的な事業内容のみ把握"
    
    @staticmethod
    def _build_enhanced_conversation_history(candidate, individual_history, all_conversations):
        """会話履歴の構築 - 自分の回答履歴と他の就活生の最新回答1件ずつを抽出"""
        
        final_history_lines = [] # 最終的な履歴行を格納するリスト
        other_candidates_added_set = set() # 他の就活生の最新発言が追加済みかを追跡するセット

        # 1. 使用する履歴ソースと最大件数を決定
        source_history = []
        max_history_length = 0
        if all_conversations and 'all' in all_conversations:
            source_history = all_conversations.get('all', [])
            max_history_length = 10 # all_conversationsがある場合は直近10件
        else:
            source_history = individual_history
            max_history_length = 6 # all_conversationsがない場合は直近6件

        # 2. 履歴を逆順に処理し、自分の発言は全て、他の就活生の発言は最新1件のみを追加
        #    面接官の発言は追加しない
        for msg in reversed(source_history):
            # 必要な件数に達したら処理を停止
            if len(final_history_lines) >= max_history_length:
                break 
            
            if msg['sender'] == candidate['name']:
                # 自分の発言は全て追加
                final_history_lines.append(f"{candidate['name']}: {msg['text']}")
            elif msg['sender'] != 'interviewer':
                # 他の就活生の発言の場合、まだその就活生の最新発言が追加されていなければ追加
                if msg['sender'] not in other_candidates_added_set:
                    final_history_lines.append(f"{msg['sender']}: {msg['text']}")
                    other_candidates_added_set.add(msg['sender'])
                # 面接官の発言はここでは追加しない

        # 3. 逆順に追加されたリストを時系列順に戻して結合し、文字列として返す
        return "\n".join(reversed(final_history_lines))
    
    @staticmethod
    def generate_interview_response(question, candidate, company, conversation_history, all_conversations=None):
        """非ストリーミング版（後方互換性のため）"""
        logger.info(f"Generating subtle difference response for {candidate['name']} (prep: {candidate['preparation']})")
        
        company_info = CompanyKnowledgeManager.get_company_info_for_candidate(
            company, candidate["preparation"], candidate["name"]
        )
        
        enhanced_history = LLamaInterviewResponseGenerator._build_enhanced_conversation_history(
            candidate, conversation_history, all_conversations
        )
        
        prompt = f"""
あなたは {candidate["name"]} という日本の就活生です。この企業に絶対に入社したく、面接官に志望度の高さを強くアピールしたいと考えています。
企業研究を熱心に行いましたが、情報収集には限界があります。知っている情報は具体的に、知らない情報は前向きな推測や一般論で補って回答してください。

# あなたのプロフィール
- 氏名: {candidate["name"]}
- 大学: {candidate["university"]}
- 強み: {candidate["strength"]}
- ガクチカ: {candidate["gakuchika"]}
- MBTI: {candidate["mbti"]}

# あなたが調べて得た企業情報（{company_info.get("knowledge_coverage")}）
{LLamaInterviewResponseGenerator._format_available_company_info(company_info)}

# 回答の重要な方針
- 非常に高い志望度と熱意を必ず示してください
- 他の就活生に負けない強い意欲を表現してください
- 知っている具体的な情報は積極的に言及してください
- 知らない情報については前向きな推測や業界一般論で補ってください
- 絶対に「知らない」「分からない」「詳しくない」とは言わないでください
- この企業で成長したい気持ちを強く表現してください

# これまでの会話
{enhanced_history}

# 面接官からの質問
{question}

---
{candidate["name"]} として、最高レベルの志望度と熱意を示しながら、150文字程度で自然な日本語で回答してください。
この企業への強い憧れと、絶対に入社したい気持ちを表現してください。
回答のみを出力し、説明や前置きは不要です。
"""
        
        from . import LLamaGenerator
        response = LLamaGenerator.generate_with_llama(prompt, max_tokens=256, temperature=0.75)
        
        response = re.sub(r'^[：:]\s*', '', response)
        response = re.sub(r'はい、承知いたしました。.*', '', response, flags=re.DOTALL)
        
        logger.info(f"Subtle difference response generated for {candidate['name']}")
        return response

# 進捗状況とデバッグログを管理するグローバル変数
generation_progress = {
    'status': 'ready', 'message': '', 'progress': 0
}

if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    logger.info("=" * 80)
    logger.info("Subtle Difference Interview Game - High/High-Middle/Middle")
    logger.info("=" * 80)
    
    if GeminiGenerator.is_available():
        logger.info("✓ Gemini API: Available (Data Generation)")
        logger.info("  - Model: models/gemini-2.5-flash-lite-preview-06-17")
    else:
        logger.warning("✗ Gemini API: Not configured")
        logger.warning("  - Falling back to local static data")
    
    initialize_model()
    logger.info("✓ Local LLama: Available (Subtle Difference Responses)")
    logger.info("✓ Game Concept: Minimal differences for maximum challenge")
    logger.info("✓ Information Distribution:")
    logger.info("  - High: 100% complete company information")
    logger.info("  - High-Middle: 85% randomly filtered information")
    logger.info("  - Middle: 70% randomly filtered information")
    logger.info("✓ Random Filtering: Different info combinations each game")
    logger.info("Game ready to start!")
    logger.info("=" * 80)

@app.route('/')
def index():
    model_available = model is not None
    gemini_available = GeminiGenerator.is_available()
    return render_template('index.html', 
                         model_available=model_available,
                         gemini_available=gemini_available)

@app.route('/api/create_game', methods=['POST'])
def create_game():
    try:
        global generation_progress, debug_logs
        
        debug_logs = []
        generation_progress = {'status': 'starting', 'message': 'ゲーム生成を準備中...', 'progress': 0}
        
        logger.info("Received request to create subtle difference interview game data.")
        
        generation_progress = {'status': 'generating_companies', 'message': '企業情報を生成中... (Gemini API)', 'progress': 10}
        companies = GeminiGenerator.generate_companies(count=1)
        # companies = GeminiGenerator._get_fallback_companies(count=1)  # フォールバックデータを使用

        generation_progress = {'status': 'generating_candidates', 'message': '就活生情報を生成中... (Gemini API)', 'progress': 50}
        candidates = GeminiGenerator.generate_candidates(count=3)
        # candidates = GeminiGenerator._get_fallback_candidates(count=3)  # フォールバックデータを使用
        
        generation_progress = {'status': 'setting_up_game', 'message': 'ゲーム設定中...', 'progress': 80}
        selected_company = random.choice(companies)
        
        # high, high-middle, middle
        prep_levels = ["high", "high-middle", "middle"]
        random.shuffle(prep_levels)
        
        for i, candidate in enumerate(candidates):
            candidate["preparation"] = prep_levels[i % len(prep_levels)]
        
        # 最も情報が少ない（middle）を正解とする
        suspicious_index = next(i for i, c in enumerate(candidates) if c["preparation"] == "middle")
        
        generation_progress = {'status': 'completed', 'message': 'ゲーム準備完了!', 'progress': 100}
        
        logger.info("Subtle difference interview game data generation completed successfully.")
        logger.info(f"Preparation levels assigned: {[c['preparation'] for c in candidates]}")
        logger.info(f"Target candidate (middle): {candidates[suspicious_index]['name']}")
        
        return jsonify({
            'status': 'success',
            'company': selected_company,
            'candidates': candidates,
            'suspicious_index': suspicious_index,
            'generation_method': 'gemini' if GeminiGenerator.is_available() else 'fallback',
            'game_type': 'subtle_difference',
            'preparation_levels': [c['preparation'] for c in candidates]
        })
        
    except Exception as e:
        logger.error(f"Subtle difference game creation error: {e}", exc_info=True)
        generation_progress = {'status': 'error', 'message': f'エラーが発生しました: {str(e)}', 'progress': 0}
        return jsonify({
            'status': 'error',
            'message': f'ゲームデータの生成中にエラーが発生しました: {str(e)}'
        }), 500

@app.route('/api/progress')
def get_progress():
    return jsonify(generation_progress)

@app.route('/api/debug_logs')
def get_debug_logs():
    return jsonify(debug_logs)

@app.route('/api/generate_stream', methods=['POST'])
def generate_stream():
    """ストリーミング応答生成"""
    try:
        data = request.json
        question = data.get('question', '')
        candidate_data = data.get('candidate', {})
        company_data = data.get('company', {})
        conversation_history = data.get('conversation_history', [])
        all_conversations = data.get('all_conversations', None)
        
        if not question:
            return jsonify({'error': 'プロンプトが空です', 'status': 'error'}), 400
        
        def generate():
            try:
                response_generator = LLamaInterviewResponseGenerator.generate_interview_response_stream(
                    question, candidate_data, company_data, conversation_history, all_conversations
                )
                
                for chunk_or_complete in response_generator:
                    if isinstance(chunk_or_complete, tuple) and chunk_or_complete[0] == "__COMPLETE__":
                        complete_response = chunk_or_complete[1]
                        yield f"data: {json.dumps({'complete_response': complete_response, 'status': 'completed'})}\n\n"
                        break
                    else:
                        yield f"data: {json.dumps({'chunk': chunk_or_complete, 'status': 'generating'})}\n\n"
                        time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Subtle difference streaming generation error: {str(e)}", exc_info=True)
                add_debug_log("STREAM_ERROR", f"Subtle difference streaming failed: {str(e)}")
                yield f"data: {json.dumps({'error': str(e), 'status': 'error'})}\n\n"
        
        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Subtle difference stream setup error: {str(e)}", exc_info=True)
        return jsonify({'error': f'ストリーミング設定中にエラーが発生しました: {str(e)}', 'status': 'error'}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        question = data.get('question', '')
        candidate_data = data.get('candidate', {})
        company_data = data.get('company', {})
        conversation_history = data.get('conversation_history', [])
        all_conversations = data.get('all_conversations', None)
        
        if not question:
            return jsonify({'error': 'プロンプトが空です', 'status': 'error'}), 400
        
        response = LLamaInterviewResponseGenerator.generate_interview_response(
            question, candidate_data, company_data, conversation_history, all_conversations
        )
        
        return jsonify({
            'response': response, 
            'status': 'success', 
            'model_used': 'llama_subtle_difference',
            'preparation_level': candidate_data.get('preparation', 'unknown'),
            'game_type': 'subtle_difference'
        })
        
    except Exception as e:
        logger.error(f"Subtle difference response generation error: {str(e)}", exc_info=True)
        add_debug_log("API_ERROR", f"Subtle difference response generation failed: {str(e)}")
        return jsonify({'error': f'応答生成中にエラーが発生しました: {str(e)}', 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        
        return jsonify({
            'status': 'healthy', 
            'model_available': True, 
            'model': MODEL_NAME,
            'gemini_available': GeminiGenerator.is_available(),
            'data_generation': 'gemini' if GeminiGenerator.is_available() else 'fallback',
            'interview_responses': 'llama_subtle_difference',
            'streaming_available': True,
            'game_features': [
                'subtle_motivation_differences_only',
                'random_information_filtering',
                'high_middle_middle_levels',
                'maximum_challenge_difficulty'
            ],
            'knowledge_levels': {
                'high': '100% complete company information',
                'high-middle': '85% randomly filtered company information', 
                'middle': '70% randomly filtered company information'
            },
            'filtering_method': 'random_per_candidate_per_game',
            'gpu_memory_allocated': f"{memory_allocated:.1f} GB",
            'gpu_memory_reserved': f"{memory_reserved:.1f} GB",
            'device': str(torch.cuda.get_device_name(0))
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("Subtle Difference Interview Game - Maximum Challenge")
    print("=" * 80)
    print(f"Local LLama Model: {MODEL_NAME}")
    
    if GeminiGenerator.is_available():
        print("✓ Data Generation: Gemini API (models/gemini-2.5-flash-lite-preview-06-17)")
        print("  - Enhanced company data with 14+ information fields")
    else:
        print("✗ Data Generation: Fallback data (Static)")
        print("  - Set GEMINI_API_KEY for faster generation")
    
    print("✓ Interview Responses: Local LLama (Subtle Differences)")
    print("✓ Game Concept:")
    print("  - ALL candidates show MAXIMUM MOTIVATION")
    print("  - Only SUBTLE information differences")
    print("  - MAXIMUM challenge difficulty")
    print("✓ Information Distribution:")
    print("  - High: 100% (all 14+ company info fields)")
    print("  - High-Middle: 85% (random 15% filtered out)")
    print("  - Middle: 70% (random 30% filtered out)")
    print("✓ Random Filtering:")
    print("  - Different combinations each game")
    print("  - Candidate-specific seed for consistency")
    print("  - Essential fields always included")
    print("✓ Real-time Streaming: ChatGPT-like experience")
    print("Challenge: Can you spot the MINIMAL differences?")
    print("Starting server...")
    print("Game Access: http://localhost:5000")
    print("Health Check: http://localhost:5000/health")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)