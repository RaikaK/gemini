"""
Hugging Face CLIを使ったローカルモデル管理システム
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from huggingface_hub import try_to_load_from_cache, snapshot_download
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceModelManager:
    """Hugging Face CLIを使ったモデル管理クラス"""
    
    def __init__(self, cache_dir: str = None):
        """
        初期化
        
        Args:
            cache_dir: モデルキャッシュディレクトリ（デフォルト: ~/.cache/huggingface/hub）
        """
        self.cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
        self.models_dir = Path(self.cache_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 利用可能なモデル情報
        self.available_models = {
            # 主要な日本語対応モデル
            "llama3": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "size_gb": 16,
                "description": "Llama 3.1 8B Instruct - 高性能な多言語モデル",
                "recommended_gpu": "RTX 4090, A100"
            },
            "ELYZA-japanese-Llama-2": {
                "model_id": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
                "size_gb": 14,
                "description": "ELYZA Japanese Llama 2 7B - 日本語特化モデル",
                "recommended_gpu": "RTX 4080, A100"
            },
            "SWALLOW": {
                "model_id": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
                "size_gb": 16,
                "description": "SWALLOW 8B - 日本語能力強化モデル",
                "recommended_gpu": "RTX 4090, A100"
            },
            "llama3-elyza-jp": {
                "model_id": "elyza/Llama-3-ELYZA-JP-8B",
                "size_gb": 16,
                "description": "Llama 3 ELYZA JP 8B - 日本語最適化モデル",
                "recommended_gpu": "RTX 4090, A100"
            },
            
            # 軽量モデル
            "calm2-3b": {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "size_gb": 2,
                "description": "TinyLlama 1.1B - 超軽量モデル（calm2-3bの代替）",
                "recommended_gpu": "RTX 3060, CPU"
            },
            "tinyllama": {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "size_gb": 2,
                "description": "TinyLlama 1.1B - 超軽量モデル",
                "recommended_gpu": "RTX 3060, CPU"
            },
            
            # 高性能モデル
            "llama3-70b": {
                "model_id": "meta-llama/Llama-3.1-70B-Instruct",
                "size_gb": 140,
                "description": "Llama 3.1 70B - 最高性能（大容量GPU必要）",
                "recommended_gpu": "A100 80GB, H100"
            },
            "mistral-7b": {
                "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "size_gb": 14,
                "description": "Mistral 7B Instruct - 高性能欧州モデル",
                "recommended_gpu": "RTX 4080, A100"
            },

            # 新しく追加されたモデル
            "qwen3-4b-instruct-2507": {
                "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                "size_gb": 8,
                "description": "Qwen3 4B Instruct 2507 - 最新世代の指示対応モデル",
                "recommended_gpu": "RTX 4070, A100"
            },
            "meta-llama-3-8b": {
                "model_id": "meta-llama/Meta-Llama-3-8B",
                "size_gb": 16,
                "description": "Meta Llama 3 8B - 最新のLlama 3モデル",
                "recommended_gpu": "RTX 4090, A100"
            },
            "qwen2.5-7b-instruct": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "size_gb": 14,
                "description": "Qwen2.5 7B Instruct - 高性能指示対応モデル",
                "recommended_gpu": "RTX 4080, A100"
            },
            "qwen3-8b": {
                "model_id": "Qwen/Qwen3-8B",
                "size_gb": 16,
                "description": "Qwen3 8B - 最新世代の汎用モデル",
                "recommended_gpu": "RTX 4090, A100"
            },
            "gemma-3-1b-it": {
                "model_id": "google/gemma-3-1b-it",
                "size_gb": 2,
                "description": "Gemma 3 1B IT - 軽量高性能モデル",
                "recommended_gpu": "RTX 3060, CPU"
            }

        }
    
    def check_hf_cli_installed(self) -> bool:
        """huggingface_hubモジュールが利用可能かをチェック"""
        try:
            import huggingface_hub  # noqa: F401
            return True
        except ImportError:
            logger.warning("huggingface_hub モジュールが見つかりません")
            return False
    
    def install_hf_cli(self) -> bool:
        """Hugging Face CLIをインストール"""
        try:
            logger.info("huggingface_hub をインストール中...")
            commands_to_try = [["uv", "pip", "install"], ["pip", "install"]]
            
            for cmd_base in commands_to_try:
                try:
                    subprocess.run(cmd_base + ["huggingface_hub[cli]"], 
                                 check=True, capture_output=True, timeout=300)
                    logger.info("huggingface_hub のインストールが完了しました")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            logger.error("huggingface_hub のインストールに失敗: pip または uv pip が見つかりません")
            return False
        except Exception as e:
            logger.error(f"huggingface_hub のインストールに失敗: {e}")
            return False
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """モデル情報を取得"""
        return self.available_models.get(model_key)
    
    def list_available_models(self) -> List[str]:
        """利用可能なモデル一覧を取得"""
        return list(self.available_models.keys())
    
    def is_model_downloaded(self, model_key: str) -> bool:
        """モデルがダウンロード済みかチェック（基本的なチェックのみ）"""
        model_info = self.get_model_info(model_key)
        if not model_info:
            return False
        
        model_path = self.models_dir / model_key
        if model_path.exists() and any(model_path.iterdir()):
            return True
        
        return False
    
    def is_model_complete(self, model_key: str) -> bool:
        """モデルが完全にダウンロードされているかチェック（ファイルの存在確認）"""
        model_path = self.get_model_path(model_key)
        if not model_path:
            return False
        
        # 必要なファイルが存在するかチェック
        # config.jsonとtokenizer関連ファイルが存在すれば基本的にOK
        required_files = ["config.json"]
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        
        has_config = (model_path / "config.json").exists()
        has_tokenizer = any((model_path / f).exists() for f in tokenizer_files)
        
        # モデルファイル（.safetensorsまたは.bin）をチェック
        model_files = [f for f in model_path.iterdir() if f.is_file() and f.suffix in [".safetensors", ".bin"]]
        has_model_file = len(model_files) > 0
        
        if not (has_config and has_tokenizer and has_model_file):
            return False
        
        # モデルファイルが分割されている場合、全てのファイルが存在するかチェック
        # model-XXXXX-of-YYYYY.safetensors の形式をチェック
        import re
        pattern = re.compile(r'model-(\d+)-of-(\d+)\.(safetensors|bin)')
        
        found_parts = {}
        for model_file in model_files:
            match = pattern.match(model_file.name)
            if match:
                part_num = int(match.group(1))
                total_parts = int(match.group(2))
                found_parts[part_num] = total_parts
        
        # 分割ファイルがある場合、全てのパートが存在するか確認
        if found_parts:
            total_parts = list(found_parts.values())[0]  # 最初のファイルから総パート数を取得
            for part_num in range(1, total_parts + 1):
                expected_name = f"model-{part_num:05d}-of-{total_parts:05d}.safetensors"
                if not (model_path / expected_name).exists():
                    # .bin形式もチェック
                    expected_name_bin = f"model-{part_num:05d}-of-{total_parts:05d}.bin"
                    if not (model_path / expected_name_bin).exists():
                        logger.warning(f"モデルファイルの一部が見つかりません: {expected_name}")
                        return False
        
        return True
    
    def download_model(self, model_key: str, force: bool = False, progress_callback=None) -> bool:
        """モデルをダウンロード（進捗表示対応）"""
        model_info = self.get_model_info(model_key)
        if not model_info:
            logger.error(f"未知のモデル: {model_key}")
            return False
        
        if self.is_model_downloaded(model_key) and not force:
            logger.info(f"モデル {model_key} は既にダウンロード済みです")
            return True
        
        model_id = model_info["model_id"]
        size_gb = model_info["size_gb"]
        
        logger.info(f"モデル {model_key} ({model_id}) をダウンロード中...")
        logger.info(f"サイズ: {size_gb}GB")
        
        if progress_callback:
            progress_callback(f"ダウンロード開始: {model_key} ({size_gb}GB)")
        
        try:
            target_dir = self.models_dir / model_key
            target_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"モデル {model_key} のダウンロードが完了しました")
            if progress_callback:
                progress_callback(f"ダウンロード完了: {model_key}")
            return True
        except Exception as e:
            logger.error(f"ダウンロード中にエラーが発生: {e}")
            if progress_callback:
                progress_callback(f"ダウンロードエラー: {e}")
            return False
    
    def get_model_path(self, model_key: str) -> Optional[Path]:
        """モデルのローカルパスを取得"""
        if not self.is_model_downloaded(model_key):
            return None
        
        model_info = self.get_model_info(model_key)
        if model_info:
            model_path = self.models_dir / model_key
            if model_path.exists() and any(model_path.iterdir()):
                return model_path
        return None
    
    def initialize_model(self, model_key: str, device: str = "auto", 
                        quantization: bool = True) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """モデルを初期化してロード"""
        model_info = self.get_model_info(model_key)
        if not model_info:
            logger.error(f"未知のモデル: {model_key}")
            return None, None
        
        model_id = model_info["model_id"]
        
        logger.info(f"モデル {model_key} ({model_id}) を初期化中...")
        
        try:
            # ローカルパスを取得（ダウンロード済みの場合）
            model_path = self.get_model_path(model_key)
            
            # モデルが完全にダウンロードされているかチェック
            is_complete = model_path and self.is_model_complete(model_key)
            
            # 使用するパスを決定
            # 完全な場合はローカルパス、不完全でもローカルパスがあればそれを使用（from_pretrainedが自動的に欠けているファイルをダウンロード）
            if model_path:
                load_path = str(model_path)
                if is_complete:
                    logger.info(f"ローカルパスからロード（完全）: {model_path}")
                else:
                    logger.warning(f"モデル {model_key} のダウンロードが不完全です。ローカルファイルからロードを試みます（欠けているファイルは自動ダウンロードされます）。")
            else:
                load_path = model_id
                logger.info(f"モデルIDから自動ダウンロード: {model_id}")
            
            # 量子化設定
            quantization_config = None
            torch_dtype = torch.float32
            
            if quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                torch_dtype = torch.bfloat16
                logger.info("4bit量子化を有効にしました")
            
            # トークナイザーをロード
            tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                trust_remote_code=True
            )
            logger.info("トークナイザーのロードが完了しました")
            
            # モデルをロード
            # from_pretrainedは自動的に欠けているファイルをダウンロードしようとします
            # ネットワークエラーの場合は、ローカルファイルのみでロードを試みる
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    trust_remote_code=True,
                    local_files_only=False  # 欠けているファイルがあればダウンロードを試みる
                )
                logger.info("モデルのロードが完了しました")
            except (RuntimeError, OSError) as e:
                error_str = str(e)
                # SSLエラーやネットワークエラーの場合
                if "SSL" in error_str or "SSLError" in error_str or "network" in error_str.lower() or "connection" in error_str.lower():
                    logger.warning(f"ネットワークエラーが発生しました: {e}")
                    if model_path:
                        logger.info("ローカルファイルのみでロードを試みます（ネットワーク接続なし）...")
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                str(model_path),
                                quantization_config=quantization_config,
                                torch_dtype=torch_dtype,
                                device_map=device,
                                trust_remote_code=True,
                                local_files_only=True  # ローカルファイルのみを使用
                            )
                            logger.info("モデルのロードが完了しました（ローカルファイルのみ）")
                        except Exception as e2:
                            logger.error(f"ローカルファイルのみでのロードも失敗しました: {e2}")
                            logger.error("モデルファイルが不完全です。ネットワーク接続を確認するか、モデルを再ダウンロードしてください。")
                            raise
                    else:
                        logger.error("ローカルファイルが存在しないため、ネットワーク接続が必要です。")
                        raise
                # ファイルが見つからない場合
                elif "No such file" in error_str or "not found" in error_str.lower():
                    if model_path:
                        logger.warning(f"ローカルファイルが見つかりません: {e}")
                        logger.info(f"モデルIDから再ダウンロードを試みます: {model_id}")
                        load_path = model_id
                        model = AutoModelForCausalLM.from_pretrained(
                            load_path,
                            quantization_config=quantization_config,
                            torch_dtype=torch_dtype,
                            device_map=device,
                            trust_remote_code=True
                        )
                        logger.info("モデルのロードが完了しました（再ダウンロード後）")
                    else:
                        raise
                else:
                    raise
            
            # パッドトークンの設定
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id
            
            logger.info(f"モデル {model_key} の初期化が完了しました")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def print_model_status(self):
        """モデル状態を表示"""
        print("\n=== ローカルモデル状態 ===")
        for model_key, model_info in self.available_models.items():
            status = "✓ ダウンロード済み" if self.is_model_downloaded(model_key) else "✗ 未ダウンロード"
            print(f"{model_key}: {status}")
            print(f"  - {model_info['description']}")
            print(f"  - サイズ: {model_info['size_gb']}GB")
            print(f"  - 推奨GPU: {model_info['recommended_gpu']}")
            print()

