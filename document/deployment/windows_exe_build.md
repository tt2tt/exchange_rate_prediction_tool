# Windows向けexeビルド手順

## 事前準備
- `python -m pip install -r requirements-dev.txt` を実行して開発依存を導入する。
- 仮想環境`venv`を有効化した状態で以下のコマンドを実行すること。

## ビルド手順
1. プロジェクトルートで `python app.py` を実行し、GUIが起動することを一度確認する。
2. 次のコマンドを実行してPyInstallerで単一ファイルexe（`--onefile`）を生成する。
   ```powershell
   cd c:/Users/grove/OneDrive/Desktop/開発/exchange_rate_prediction_tool
   .\venv\Scripts\python.exe scripts/build_windows_exe.py
   ```
3. ビルド完了後、`dist/exchange_rate_prediction_tool.exe` が生成される。
4. exeを実行してGUIが問題なく表示されるか確認する。

## 補足
- PyInstaller実行時に `build/` と `dist/` ディレクトリ、`app.spec` が作成される。不要になった場合は削除してよい。
- 起動時にDLL不足等のエラーが出た場合は、配布先PCにもMicrosoft Visual C++再頒布可能パッケージがインストールされているか確認する。
