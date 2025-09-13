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
    
    def initialize_spreadsheet(self) -> Dict[str, Any]:
        """
        スプレッドシートを初期化（ヘッダー行を設定）
        
        Returns:
            Dict[str, Any]: 初期化結果
        """
        try:
            payload = {'action': 'initialize'}
            response = self.session.post(self.gas_web_app_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info("スプレッドシートの初期化に成功しました")
            else:
                logger.error(f"スプレッドシート初期化エラー: {result.get('message')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"スプレッドシート初期化エラー: {e}")
            return {
                'success': False,
                'message': f'初期化エラー: {str(e)}'
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
    
    def record_multiple_experiment_results(self, experiment_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        複数の実験結果をスプレッドシートに一括記録
        
        Args:
            experiment_data_list (List[Dict[str, Any]]): 実験結果データのリスト
            
        Returns:
            Dict[str, Any]: 記録結果
        """
        try:
            payload = {
                'action': 'record_multiple',
                'data': experiment_data_list
            }
            
            response = self.session.post(self.gas_web_app_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info(f"{len(experiment_data_list)}件の実験結果の記録に成功しました")
            else:
                logger.error(f"実験結果一括記録エラー: {result.get('message')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"実験結果一括記録エラー: {e}")
            return {
                'success': False,
                'message': f'一括記録エラー: {str(e)}'
            }
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return {
                'success': False,
                'message': f'予期しないエラー: {str(e)}'
            }
    
    def get_spreadsheet_stats(self) -> Dict[str, Any]:
        """
        スプレッドシートの統計情報を取得
        
        Returns:
            Dict[str, Any]: 統計情報
        """
        try:
            response = self.session.get(f"{self.gas_web_app_url}?action=stats")
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info("スプレッドシート統計情報の取得に成功しました")
            else:
                logger.error(f"統計情報取得エラー: {result.get('message')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {
                'success': False,
                'message': f'統計情報取得エラー: {str(e)}'
            }
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return {
                'success': False,
                'message': f'予期しないエラー: {str(e)}'
            }
    
    def clear_spreadsheet_data(self) -> Dict[str, Any]:
        """
        スプレッドシートのデータをクリア（ヘッダー行以外を削除）
        
        Returns:
            Dict[str, Any]: クリア結果
        """
        try:
            payload = {'action': 'clear'}
            response = self.session.post(self.gas_web_app_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info("スプレッドシートデータのクリアに成功しました")
            else:
                logger.error(f"データクリアエラー: {result.get('message')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"データクリアエラー: {e}")
            return {
                'success': False,
                'message': f'データクリアエラー: {str(e)}'
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

def create_spreadsheet_config_template():
    """
    スプレッドシート設定ファイルのテンプレートを作成
    """
    config_file = Path(__file__).parent / 'spreadsheet_config.json'
    
    template = {
        "gas_web_app_url": "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec",
        "description": "Google Apps Script WebアプリのURLを設定してください",
        "setup_instructions": [
            "1. Google Apps Script エディタで新しいプロジェクトを作成",
            "2. gas_spreadsheet_integration.js のコードをコピー&ペースト",
            "3. スプレッドシートのIDを設定",
            "4. デプロイしてWebアプリとして公開",
            "5. WebアプリのURLをここに設定"
        ]
    }
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        logger.info(f"スプレッドシート設定テンプレートを作成しました: {config_file}")
        return True
        
    except Exception as e:
        logger.error(f"設定テンプレート作成エラー: {e}")
        return False

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

# 使用例
if __name__ == "__main__":
    # 設定テンプレートを作成
    create_spreadsheet_config_template()
    
    # スプレッドシート連携をテスト
    integration = get_spreadsheet_integration()
    
    if integration:
        # 接続テスト
        result = integration.test_connection()
        print(f"接続テスト結果: {result}")
        
        # 統計情報取得
        stats = integration.get_spreadsheet_stats()
        print(f"統計情報: {stats}")
    else:
        print("スプレッドシート連携が設定されていません")
