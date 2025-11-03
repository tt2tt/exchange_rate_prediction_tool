"""PyInstallerでWindows向け実行ファイルを生成する補助スクリプト。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _remove_path(target_path: Path) -> None:
    """既存のビルド成果物を安全に削除する。

    Args:
        target_path: 削除対象となるパス。
    """
    # 対象が存在する場合のみ削除を実施する
    if not target_path.exists():
        return
    # ディレクトリは中身ごと削除し、ファイルは単体で削除する
    if target_path.is_dir():
        shutil.rmtree(target_path)
    else:
        target_path.unlink()


def main() -> None:
    """PyInstallerを呼び出してGUIアプリの単体実行ファイルを生成する。

    Raises:
        subprocess.CalledProcessError: PyInstallerの実行が失敗した場合。
    """
    # プロジェクトルートを起点に各種パスを決定する
    project_root = Path(__file__).resolve().parents[1]
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    spec_file = project_root / "app.spec"

    # 前回成果物を削除してクリーンビルドを担保する
    for target in (dist_dir, build_dir, spec_file):
        _remove_path(target)

    # PyInstallerに渡す引数を定義してビルド条件を固定する
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "exchange_rate_prediction_tool",
        "--noconsole",
        str(project_root / "app.py"),
    ]

    # サブプロセスとしてPyInstallerを起動しビルドを実行する
    subprocess.run(command, check=True, cwd=project_root)
    # 成果物の格納先を標準出力に表示して利用者へ案内する
    print(f"ビルドが完了しました。distフォルダ内を確認してください: {dist_dir}")


if __name__ == "__main__":
    main()
