#!/usr/bin/env python3
import argparse
import json
import os
import time
import sys
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class MLXProcessor:
    """MLXを使ってLLMを処理するクラス"""
    
    def __init__(self, config_path="template_config.json"):
        """初期化"""
        # 設定を読み込む
        self.config = self._load_config(config_path)
        
        # 設定から各種パラメータを取得
        self.mlx_settings = self.config.get("mlx_settings", {})
        self.script_settings = self.config.get("script_settings", {})
        
        # MLX LMのimportをここで行う
        try:
            from mlx_lm import load, generate
            self.load = load
            self.generate = generate
            
            # ストリーミング生成が必要な場合
            from mlx_lm import stream_generate
            self.stream_generate = stream_generate
        except ImportError:
            print("エラー: mlx-lm パッケージがインストールされていません。")
            print("インストール方法: pip install mlx-lm")
            sys.exit(1)
        
        # モデルとトークナイザーをキャッシュするための変数
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 設定ファイル {config_path} が見つかりません。デフォルト設定を使用します。")
            return {
                "mlx_settings": {
                    "request_options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_tokens": 2048
                    }
                },
                "script_settings": {
                    "batch_size": 10,
                    "retry_attempts": 3,
                    "timeout": 120
                },
                "input_format": {
                    "required_fields": ["id", "role", "text"]
                },
                "output_format": {
                    "fields": ["id", "role", "text"]
                },
                "multi_turn_conversation": {
                    "enabled": False
                }
            }
        except json.JSONDecodeError:
            print(f"エラー: 設定ファイル {config_path} の形式が不正です。")
            sys.exit(1)
    
    def read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """指定されたJSONLファイルを読み込む"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line_num, line in enumerate(tqdm(lines, desc="JSONLファイルを読み込み中"), 1):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"警告: {line_num}行目のJSON形式が不正です。スキップします。")
        except FileNotFoundError:
            print(f"エラー: 入力ファイル {file_path} が見つかりません。")
            sys.exit(1)
        
        return data
    
    def write_jsonl(self, file_path: str, data: List[Dict[str, Any]]) -> None:
        """指定されたデータをJSONLファイルに書き込む"""
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in tqdm(data, desc="結果を保存中"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def validate_input(self, item: Dict[str, Any]) -> bool:
        """入力データの形式が正しいか検証する"""
        required_fields = self.config.get("input_format", {}).get("required_fields", ["id", "role", "text"])
        
        # 必須フィールドの存在確認
        for field in required_fields:
            if field not in item:
                return False
        
        # roleとtextが配列形式であることを確認
        if not isinstance(item.get('role'), list) or not item.get('role'):
            return False
        
        if not isinstance(item.get('text'), list) or not item.get('text'):
            return False
        
        return True
    
    def load_model(self, model_name: str) -> bool:
        """指定されたモデルをロードする"""
        # すでに同じモデルがロードされている場合はスキップ
        if self.model is not None and self.current_model_name == model_name:
            return True
        
        try:
            print(f"モデル '{model_name}' をロード中...")
            
            # トークナイザーの設定パラメータを準備
            tokenizer_config = {
                "trust_remote_code": True  # リモートコードを信頼する設定
            }
            
            # モデルをロード（Hugging Faceリポジトリ名またはローカルパスを指定）
            try:
                self.model, self.tokenizer = self.load(
                    model_name,
                    tokenizer_config=tokenizer_config
                )
            except Exception as load_error:
                print(f"基本的なロード方法でエラーが発生しました: {load_error}")
                # 追加のオプションを試す
                print("代替ロード方法を試みます...")
                try:
                    self.model, self.tokenizer = self.load(model_name)
                except Exception as fallback_error:
                    print(f"代替ロード方法でもエラーが発生しました: {fallback_error}")
                    raise
            
            # モデルのAPIバージョンをチェック
            self._check_api_version()
            
            self.current_model_name = model_name
            print(f"モデル '{model_name}' のロードが完了しました。")
            return True
        except Exception as e:
            print(f"エラー: モデル '{model_name}' のロード中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_api_version(self):
        """使用中のMLX-LMのAPIバージョンを確認し、適切な生成方法を決定"""
        try:
            # APIの詳細情報を取得
            import inspect
            sig = inspect.signature(self.generate)
            params = sig.parameters
            
            # APIのパラメータを確認
            param_names = set(params.keys())
            print(f"INFO: MLX-LM generate関数のパラメータ: {param_names}")
            
            # APIバージョンを設定 (mlx-lmのバージョンにより異なる)
            # 新しいバージョンの場合: sampler引数がある
            has_sampler = 'sampler' in param_names
            # 古いバージョンの場合: temperature引数がある
            has_temperature = 'temperature' in param_names
            # ストリーミング向け
            has_formatter = 'formatter' in param_names
            
            if has_sampler:
                self.api_version = "new_api"
                print("INFO: MLX-LM 最新API (sampler引数) を使用します")
            elif has_temperature:
                self.api_version = "old_api"
                print("INFO: MLX-LM 旧API (temperature引数) を使用します")
            else:
                self.api_version = "minimal_api"
                print("INFO: MLX-LM 最小限のAPIを使用します")
            
            # サンプラーユーティリティをインポートを試みる
            try:
                # サンプラー作成関数をインポート
                from mlx_lm.sample_utils import make_sampler
                self.make_sampler = make_sampler
                print("INFO: make_sampler関数を正常にインポートしました")
            except ImportError:
                self.make_sampler = None
                print("INFO: make_sampler関数をインポートできませんでした")
                
        except Exception as e:
            print(f"警告: APIバージョン確認中にエラーが発生しました: {e}")
            self.api_version = "minimal_api"
    
    def process_item(self, model_name: str, item: Dict[str, Any], retry_attempts: int) -> Optional[Dict[str, Any]]:
        """MLXを使用して単一のアイテムを処理する"""
        if not self.validate_input(item):
            print(f"警告: 無効な形式のアイテムをスキップします: {item}")
            return None
        
        # リクエストオプションを取得
        request_options = self.mlx_settings.get("request_options", {})
        
        # マルチターン会話の有効/無効を確認
        multi_turn_enabled = self.config.get("multi_turn_conversation", {}).get("enabled", False)
        
        # リトライロジック
        for attempt in range(retry_attempts):
            try:
                if multi_turn_enabled and len(item['role']) > 1:
                    # マルチターン会話の場合
                    messages = []
                    for i, role in enumerate(item['role']):
                        if i < len(item['text']):
                            # チャットテンプレート用にroleを変換
                            chat_role = "assistant" if role != "user" else "user"
                            messages.append({
                                'role': chat_role,
                                'content': item['text'][i]
                            })
                    
                    # 最後のメッセージがユーザーからでない場合は処理をスキップ
                    if not messages or messages[-1]['role'] != "user":
                        print(f"警告: ID {item.get('id', '不明')} の最後のメッセージがユーザーからではありません。スキップします。")
                        return None
                    
                    # チャットテンプレートを適用
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True
                    )
                else:
                    # 単一ターンの場合
                    prompt = item['text'][0]
                
                # MLXでテキスト生成
                # サンプリングパラメータを辞書で作成
                sampling_params = {
                    "max_tokens": request_options.get("max_tokens", 2048),
                    "verbose": False
                }
                
                # 以下のパラメータはsamplerに渡す準備をする
                temp = request_options.get("temperature", 0.7)
                top_p = request_options.get("top_p", 0.9)
                top_k = request_options.get("top_k", 40)
                
                # APIバージョンに応じた呼び出し方法を選択
                if hasattr(self, 'api_version'):
                    if self.api_version == "new_api":
                        # 最新のAPIパターン（sampler引数を使用）
                        if self.make_sampler:
                            # make_samplerが利用可能な場合はそれを使用
                            sampler = self.make_sampler(
                                temp, 
                                top_p, 
                                min_p=0.0, 
                                min_tokens_to_keep=1
                            )
                        else:
                            # 基本的なサンプリング関数を定義
                            def basic_sampler(logits):
                                # シンプルな温度スケーリングと確率サンプリング
                                import numpy as np
                                if temp == 0:
                                    return np.argmax(logits, axis=-1)
                                logits = logits / max(temp, 1e-5)
                                probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                                return np.random.choice(probs.shape[-1], p=probs.flatten())
                            sampler = basic_sampler
                        
                        # 最新のAPIパターンで呼び出し
                        model_response = self.generate(
                            self.model,
                            self.tokenizer,
                            prompt=prompt,
                            max_tokens=request_options.get("max_tokens", 2048),
                            sampler=sampler,
                            verbose=False
                        )
                    elif self.api_version == "old_api":
                        # 旧APIパターン（temperature引数を使用）
                        model_response = self.generate(
                            self.model,
                            self.tokenizer,
                            prompt=prompt,
                            max_tokens=request_options.get("max_tokens", 2048),
                            temperature=request_options.get("temperature", 0.7),
                            top_p=request_options.get("top_p", 0.9),
                            verbose=False
                        )
                    else:
                        # 最小限のAPIパターン（引数を最小限に）
                        model_response = self.generate(
                            self.model,
                            self.tokenizer,
                            prompt=prompt,
                            verbose=False
                        )
                else:
                    # APIバージョンが特定できない場合の最小限の呼び出し
                    print("INFO: 最小限のAPIで試行します...")
                    model_response = self.generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt
                    )
                
                # 結果を適切な形式で保存
                result = {
                    'id': item['id'],
                    'role': item['role'] + [model_name],
                    'text': item['text'] + [model_response]
                }
                
                return result
                
            except Exception as e:
                print(f"エラー (試行 {attempt+1}/{retry_attempts}): ID {item.get('id', '不明')} の処理中にエラーが発生しました: {e}")
                if attempt < retry_attempts - 1:
                    # リトライ前に少し待機
                    time.sleep(2)
        
        print(f"警告: ID {item.get('id', '不明')} の最大リトライ回数に達しました。スキップします。")
        return None
    
    def process_batch(self, model_name: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチ処理を行う"""
        results = []
        retry_attempts = self.script_settings.get("retry_attempts", 3)
        
        for item in tqdm(batch, desc="バッチ処理中"):
            result = self.process_item(model_name, item, retry_attempts)
            if result:
                results.append(result)
        
        return results
    
    def process_all(self, model_name: str, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """全データをバッチに分けて処理する"""
        results = []
        batch_size = self.script_settings.get("batch_size", 10)
        
        # 全体の進行状況を表示
        total_batches = (len(input_data) + batch_size - 1) // batch_size
        with tqdm(total=len(input_data), desc=f"モデル {model_name} で処理中") as pbar:
            # データをバッチに分割して処理
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i+batch_size]
                batch_results = self.process_batch(model_name, batch)
                results.extend(batch_results)
                pbar.update(len(batch))
        
        return results
    
    def run(self, model_name: str, input_path: str, output_path: str) -> None:
        """メイン処理を実行する"""
        # モデルをロード
        if not self.load_model(model_name):
            print(f"エラー: モデル '{model_name}' をロードできませんでした。")
            sys.exit(1)
        
        # 入力ファイルを読み込む
        print(f"入力ファイル {input_path} を読み込んでいます...")
        input_data = self.read_jsonl(input_path)
        print(f"{len(input_data)}件のデータを読み込みました")
        
        if not input_data:
            print("警告: 入力データが空です。処理を終了します。")
            sys.exit(0)
        
        # MLXで処理
        results = self.process_all(model_name, input_data)
        print(f"{len(results)}/{len(input_data)}件のデータの処理が完了しました")
        
        # 結果を保存
        print(f"結果を {output_path} に保存しています...")
        self.write_jsonl(output_path, results)
        print("処理が完了しました！")


def main():
    parser = argparse.ArgumentParser(description='MLX-LMを使用して入力テキストを処理するスクリプト')
    parser.add_argument('--model', type=str, required=True, help='使用するモデル名またはHugging Faceリポジトリ')
    parser.add_argument('--input', type=str, required=True, help='入力JSONLファイルのパス')
    parser.add_argument('--output', type=str, required=True, help='出力JSONLファイルのパス')
    parser.add_argument('--config', type=str, default='template_config.json', help='設定ファイルのパス（デフォルト: template_config.json）')
    
    args = parser.parse_args()
    
    # プロセッサを初期化して実行
    processor = MLXProcessor(args.config)
    processor.run(args.model, args.input, args.output)


if __name__ == "__main__":
    main()