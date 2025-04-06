MLX_Inference
Apple SiliconデバイスでMLX-LMを使用した高速LLM推論ツール
概要
MLX_Inferenceは、Appleシリコン（M1/M2/M3）搭載のMacで高速にLLMを実行するためのツールです。MLX（Apple Machine Learning）フレームワークをベースにしたMLX-LMを使用し、JSONLファイル形式のプロンプトを効率的に処理します。
特徴

Apple Silicon向けに最適化された高速推論
Hugging Faceのモデルを直接利用可能
シングルターンおよびマルチターン会話のサポート
バッチ処理による効率的な推論
量子化モデル（4bit/8bit）との互換性
柔軟な設定オプション

ディレクトリ構成
コピーMLX_Inference/
├── config/
│   └── template_config.json   # 設定テンプレート
├── example/
│   ├── multi_turn_example.jsonl    # マルチターン会話の例
│   └── single_turn_example.jsonl   # シングルターン会話の例
├── results/                  # 出力結果の保存先
├── script/
│   └── inference_mlx.py      # メインスクリプト
├── README.md                 # このファイル
└── requirements.txt          # 必要なパッケージ一覧
必要条件

Apple Silicon搭載のMac（M1/M2/M3チップ）
Python 3.8以上
必要なパッケージ（requirements.txtを参照）

インストール
bashコピー# リポジトリをクローン
git clone https://github.com/yourusername/MLX_Inference.git
cd MLX_Inference

# 仮想環境の作成と有効化（推奨）
python -m venv venv
source venv/bin/activate

# 必要なパッケージのインストール
pip install -r requirements.txt
使い方
基本的な実行方法
bashコピーpython script/inference_mlx.py --model <モデル名> --input <入力ファイル> --output <出力ファイル>
例
bashコピー# 基本的な使用例
python script/inference_mlx.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --input example/single_turn_example.jsonl --output results/single_turn_results.jsonl

# マルチターン会話の処理
python script/inference_mlx.py --model mlx-community/Llama-3-8B-Chat-4bit --input example/multi_turn_example.jsonl --output results/multi_turn_results.jsonl --config config/custom_config.json
コマンドライン引数
引数説明必須例--model使用するモデル名またはHugging Faceリポジトリはいmlx-community/Mistral-7B-Instruct-v0.3-4bit--input入力JSONLファイルのパスはいexample/single_turn_example.jsonl--output出力JSONLファイルのパスはいresults/output.jsonl--config設定ファイルのパスいいえconfig/template_config.json
設定ファイル
config/template_config.jsonで各種パラメータを設定できます：
jsonコピー{
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
  "multi_turn_conversation": {
    "enabled": false
  }
}
入出力形式
入力JSONL形式
シングルターン：
jsonコピー{"id": 1, "role": ["user"], "text": ["こんにちは、今日の天気を教えてください"]}
マルチターン：
jsonコピー{"id": 2, "role": ["user", "assistant", "user"], "text": ["こんにちは", "こんにちは！何かお手伝いできることはありますか？", "MLXフレームワークについて教えてください"]}
出力JSONL形式
jsonコピー{"id": 1, "role": ["user", "mistral-7b"], "text": ["こんにちは、今日の天気を教えてください", "申し訳ありませんが、私は特定の地域の現在の天気情報にアクセスできません。..."]}
サポートされているモデル
MLX-LMは、Hugging Face Hubの多数のモデルと互換性があります：
推奨モデル

mlx-community/Mistral-7B-Instruct-v0.3-4bit - 4ビット量子化Mistral 7B
mlx-community/Llama-3-8B-Chat-4bit - 4ビット量子化Llama 3 8B
mlx-community/phi-3-mini-4k-8bit - 8ビット量子化Phi-3-mini
mlx-community/Llama-3.2-3B-Instruct-4bit - Llama 3.2 3B (軽量かつ高性能)

より多くのモデルは [Hugging Face MLX Community](https://huggingface.co/mlx-community) で見つけることができます。