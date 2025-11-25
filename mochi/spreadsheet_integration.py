"""
スプレッドシート連携モジュール
Google Apps Script (GAS) を使用してスプレッドシートに実験結果を記録する機能
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpreadsheetIntegration:
    """スプレッドシート連携クラス"""
    
    def __init__(self, gas_web_app_url: str):
        """
        初期化
        
        Args:
            gas_web_app_url (str): Google Apps Script WebアプリのURL
        """
        self.gas_web_app_url = gas_web_app_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'InterviewSimulation/1.0'
        })
    
    def test_connection(self) -> Dict[str, Any]:
        """
        スプレッドシートとの接続をテスト
        
        Returns:
            Dict[str, Any]: テスト結果
        """
        try:
            response = self.session.get(f"{self.gas_web_app_url}?action=stats")
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info("スプレッドシートとの接続に成功しました")
                return {
                    'success': True,
                    'message': '接続成功',
                    'stats': result.get('stats', {})
                }
            else:
                logger.error(f"スプレッドシート接続エラー: {result.get('message')}")
                return {
                    'success': False,
                    'message': result.get('message', '不明なエラー')
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"スプレッドシート接続エラー: {e}")
            return {
                'success': False,
                'message': f'接続エラー: {str(e)}'
            }
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return {
                'success': False,
                'message': f'予期しないエラー: {str(e)}'
            }
    
    def record_experiment_result(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        単一の実験結果をスプレッドシートに記録
        
        Args:
            experiment_data (Dict[str, Any]): 実験結果データ
            
        Returns:
            Dict[str, Any]: 記録結果
        """
        try:
            payload = {
                'action': 'record',
                'data': experiment_data
            }
            
            response = self.session.post(self.gas_web_app_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info(f"実験結果の記録に成功しました (行: {result.get('row', 'N/A')})")
            else:
                logger.error(f"実験結果記録エラー: {result.get('message')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"実験結果記録エラー: {e}")
            return {
                'success': False,
                'message': f'記録エラー: {str(e)}'
            }
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return {
                'success': False,
                'message': f'予期しないエラー: {str(e)}'
            }

def load_spreadsheet_config() -> Optional[Dict[str, str]]:
    """
    スプレッドシート設定を読み込み
    
    Returns:
        Optional[Dict[str, str]]: 設定情報（GAS WebアプリURL等）
    """
    config_file = Path(__file__).parent / 'spreadsheet_config.json'
    
    if not config_file.exists():
        logger.warning("スプレッドシート設定ファイルが見つかりません")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'gas_web_app_url' not in config:
            logger.error("スプレッドシート設定にgas_web_app_urlが設定されていません")
            return None
        
        return config
        
    except Exception as e:
        logger.error(f"スプレッドシート設定の読み込みエラー: {e}")
        return None

def get_spreadsheet_integration() -> Optional[SpreadsheetIntegration]:
    """
    スプレッドシート連携インスタンスを取得
    
    Returns:
        Optional[SpreadsheetIntegration]: スプレッドシート連携インスタンス
    """
    config = load_spreadsheet_config()
    
    if not config:
        logger.warning("スプレッドシート設定が読み込めません。連携機能は無効です。")
        return None
    
    return SpreadsheetIntegration(config['gas_web_app_url'])

