"""LogManagerの動作検証。"""

from datetime import datetime
from pathlib import Path

from loguru import logger

from core.logging import LogManager


def test_log_manager_outputs_log(tmp_path: Path) -> None:
    """ログ出力先に日付付きファイルが生成されることを確認する。"""
    log_dir = tmp_path / "logs"
    manager = LogManager(log_dir=log_dir, retention_days=30)

    bound_logger = manager.get_logger("test")
    bound_logger.info("ログ出力テスト")

    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"run_{today}.log"
    assert log_file.exists(), "ログファイルが生成されていません"

    contents = log_file.read_text(encoding="utf-8")
    assert "ログ出力テスト" in contents

    # 設定が複数回呼ばれても冪等であることを確認
    second_logger = manager.get_logger()
    assert second_logger is logger