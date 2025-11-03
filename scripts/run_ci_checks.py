"""CIでテストと型チェックをまとめて実行するスクリプト。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_command(command: list[str]) -> None:
    """指定されたコマンドをサブプロセスで実行する。

    Args:
        command: 実行したいコマンドと引数のリスト。

    Raises:
        subprocess.CalledProcessError: コマンドの実行が失敗した場合。
    """
    # 実行内容をログとして出力し、CIの履歴から確認できるようにする
    print(f"[RUN] {' '.join(command)}")
    # リポジトリルートをカレントディレクトリに固定してコマンドを実行する
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        command,
        check=True,
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": str(project_root)},
    )


def main() -> None:
    """テスト・カバレッジ・型チェックを順番に実行する。"""
    # pytestでユニットテストとUIテストを実行し、カバレッジを算出する
    _run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=core",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-fail-under=90",
        ]
    )
    # mypyで型チェックを実行し、型安全性を担保する
    _run_command([sys.executable, "-m", "mypy", "."])


if __name__ == "__main__":
    main()
