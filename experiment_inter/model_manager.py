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
from huggingface_hub import try_to_load_from_cache
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
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
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
        """Hugging Face CLIがインストールされているかチェック"""
        # 新しいhfコマンドを優先的に試す
        commands_to_try = ["hf", "huggingface-cli"]
        
        for cmd in commands_to_try:
            try:
                # versionコマンドを使用（--versionではなく）
                result = subprocess.run([cmd, "version"], 
                                      capture_output=True, text=True, timeout=10)
                
                # デバッグ情報をログに出力
                logger.info(f"Command: {cmd} version")
                logger.info(f"Return code: {result.returncode}")
                logger.info(f"Stdout: {result.stdout}")
                logger.info(f"Stderr: {result.stderr}")
                
                # returncodeが0で、かつ標準出力または標準エラーにバージョン情報が含まれている場合
                if result.returncode == 0:
                    # バージョン情報が出力されているかチェック
                    output = result.stdout + result.stderr
                    if "version" in output.lower() or "huggingface" in output.lower():
                        logger.info(f"Hugging Face CLI detected with command: {cmd}")
                        return True
                        
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.info(f"Command {cmd} version failed: {e}")
                continue
        
        logger.warning("Hugging Face CLI not detected with any command")
        return False
    
    def install_hf_cli(self) -> bool:
        """Hugging Face CLIをインストール"""
        try:
            logger.info("Hugging Face CLIをインストール中...")
            subprocess.run(["pip", "install", "huggingface_hub[cli]"], 
                         check=True, capture_output=True)
            logger.info("Hugging Face CLIのインストールが完了しました")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Hugging Face CLIのインストールに失敗: {e}")
            return False
    
    def login_hf(self) -> bool:
        """Hugging Faceにログイン"""
        try:
            logger.info("Hugging Faceにログイン中...")
            result = subprocess.run(["huggingface-cli", "login"], 
                                  input="", text=True, timeout=60)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("ログインがタイムアウトしました")
            return False
        except Exception as e:
            logger.error(f"ログインエラー: {e}")
            return False
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """モデル情報を取得"""
        return self.available_models.get(model_key)
    
    def list_available_models(self) -> List[str]:
        """利用可能なモデル一覧を取得"""
        return list(self.available_models.keys())
    
    def is_model_downloaded(self, model_key: str) -> bool:
        """モデルがダウンロード済みかチェック"""
        model_info = self.get_model_info(model_key)
        if not model_info:
            return False
        
        # まず、model_keyで直接チェック（ダウンロード時に使用されるパス）
        model_path = self.models_dir / model_key
        if model_path.exists() and any(model_path.iterdir()):
            return True
        
        # 次に、Hugging Faceの標準キャッシュディレクトリ構造でチェック
        model_id = model_info["model_id"]
        hf_model_path = self.models_dir / "--".join(model_id.split("/"))
        if hf_model_path.exists() and any(hf_model_path.iterdir()):
            return True
        
        return False
    
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
            # huggingface-cli downloadコマンドを使用
            cmd = ["huggingface-cli", "download", model_id, "--local-dir", 
                   str(self.models_dir / model_key)]
            
            if force:
                cmd.append("--force")
            
            # 進捗表示付きでダウンロード実行
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1, universal_newlines=True)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"ダウンロード進捗: {output.strip()}")
                    if progress_callback:
                        progress_callback(f"ダウンロード中: {output.strip()}")
            
            return_code = process.poll()
            
            if return_code == 0:
                logger.info(f"モデル {model_key} のダウンロードが完了しました")
                if progress_callback:
                    progress_callback(f"ダウンロード完了: {model_key}")
                return True
            else:
                logger.error(f"ダウンロードエラー: 終了コード {return_code}")
                if progress_callback:
                    progress_callback(f"ダウンロードエラー: {model_key}")
                return False
                
        except Exception as e:
            logger.error(f"ダウンロード中にエラーが発生: {e}")
            if progress_callback:
                progress_callback(f"ダウンロードエラー: {e}")
            return False
    
    def get_model_path(self, model_key: str) -> Optional[Path]:
        """モデルのローカルパスを取得"""
        if not self.is_model_downloaded(model_key):
            return None
        
        # まず、model_keyで直接チェック
        model_path = self.models_dir / model_key
        if model_path.exists() and any(model_path.iterdir()):
            return model_path
        
        # 次に、Hugging Faceの標準キャッシュディレクトリ構造でチェック
        model_info = self.get_model_info(model_key)
        if model_info:
            model_id = model_info["model_id"]
            hf_model_path = self.models_dir / "--".join(model_id.split("/"))
            if hf_model_path.exists() and any(hf_model_path.iterdir()):
                return hf_model_path
        
        return None
    
    def initialize_model(self, model_key: str, device: str = "auto", 
                        quantization: bool = True) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """モデルを初期化してロード"""
        model_info = self.get_model_info(model_key)
        if not model_info:
            logger.error(f"未知のモデル: {model_key}")
            return None, None
        
        model_path = self.get_model_path(model_key)
        if not model_path:
            logger.error(f"モデル {model_key} がダウンロードされていません")
            return None, None
        
        logger.info(f"モデル {model_key} を初期化中...")
        
        try:
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
                str(model_path),
                trust_remote_code=True
            )
            logger.info("トークナイザーのロードが完了しました")
            
            # モデルをロード
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True
            )
            logger.info("モデルのロードが完了しました")
            
            # パッドトークンの設定
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id
            
            logger.info(f"モデル {model_key} の初期化が完了しました")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            return None, None
    
    def cleanup_model(self, model_key: str) -> bool:
        """モデルを削除"""
        model_path = self.get_model_path(model_key)
        if not model_path:
            return True
        
        try:
            shutil.rmtree(model_path)
            logger.info(f"モデル {model_key} を削除しました")
            return True
        except Exception as e:
            logger.error(f"モデル削除エラー: {e}")
            return False
    
    def get_disk_usage(self) -> Dict[str, int]:
        """ディスク使用量を取得"""
        usage = {}
        for model_key in self.available_models.keys():
            model_path = self.get_model_path(model_key)
            if model_path and model_path.exists():
                try:
                    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    usage[model_key] = total_size
                except Exception:
                    usage[model_key] = 0
        return usage
    
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
        
        # ディスク使用量
        usage = self.get_disk_usage()
        if usage:
            print("=== ディスク使用量 ===")
            total_size = 0
            for model_key, size in usage.items():
                size_gb = size / (1024**3)
                total_size += size_gb
                print(f"{model_key}: {size_gb:.1f}GB")
            print(f"合計: {total_size:.1f}GB")
