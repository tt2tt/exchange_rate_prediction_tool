# 為替予測ツール（PySide6）

## プロジェクト概要
本プロジェクトは、PySide6 を用いた Windows 向け為替予想ツールを個人利用目的で開発しています。指定の CSV データを読み込み、ロジスティック回帰モデルで次の 1 本先の終値が上昇するか下落するかを二値分類します。UI 上でスプレッド設定を変更しながら、学習・評価・バックテスト・ミニチャート可視化までをワンクリックで実行できます。配布予定はなく、開発者自身の検証用途を想定しています。

## 主な機能
- CSV 読み込みと必須カラム検証、日付ソート、欠損/無限値チェック
- 特徴量生成（リターン、移動平均乖離率、傾き、クロスフラグ、ボラティリティなど）
- ロジスティック回帰による学習と TimeSeriesSplit 検証
- 期待値シミュレーションを含む指標出力（Accuracy、Precision、Recall、ROC-AUC 等）
- スプレッド考慮付きバックテストとトレードログ CSV 出力
- PySide6 UI でのファイル選択、進捗表示、結果サマリー表示
- 終値・移動平均・シグナルを描画するミニチャート表示
- 日次ローテーション対応ログ出力（保持期間 30 日）
- 学習済みモデルの保存（`models/latest_model.joblib`）

## 想定利用者
個人用途の検証ツールとして開発者本人が利用することを前提としています。現時点で第三者への配布計画はありません。

## 開発環境
- OS: Windows 11
- 言語・フレームワーク: Python 3.11 以上、PySide6、pandas、scikit-learn など
- 仮想環境: `python -m venv venv`

## セットアップ手順
1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. （開発時）`pip install -r requirements-dev.txt`

## 実行方法
GUI の利用手順や生成物の詳細は `USAGE.md` を参照してください。ソースコードから起動する場合は仮想環境を有効化した上で `python app.py` を実行します。個人利用のため配布用インストーラーは提供していませんが、必要に応じて `scripts/build_windows_exe.py` でローカルビルドが可能です。

## テストと品質管理
- `python scripts/run_ci_checks.py` で pytest（カバレッジ取得）と mypy を一括実行
- GitHub Actions（`.github/workflows/ci.yml`）でテスト・型チェックと PyInstaller ビルドを自動化（開発者の検証用）
- 進捗とドキュメントは `document/` 配下で管理（`roadmap.md`, `progress.txt`, `operations/operations_guide.md`）

## フォルダ構成
- `app.py` ... アプリケーションエントリーポイント
- `ui/` ... PySide6 UI 実装
- `core/` ... データ前処理・特徴量生成・モデル学習・バックテスト
- `scripts/` ... CI チェックや PyInstaller ビルドスクリプト
- `tests/` ... ユニットテスト・統合テスト
- `logs/`, `models/`, `backtests/` ... 実行時生成物
- `document/` ... ロードマップ・運用ガイド等

## ライセンス
個人利用のため公開ライセンスは設定していません。外部公開の予定が生じた場合は改めて方針を定めます。