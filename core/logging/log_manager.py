"""ログ出力設定を司るマネージャークラス。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


class LogManager:
    """アプリ全体のログ設定を管理する。"""

    def __init__(self, log_dir: Path | str = Path("logs"), retention_days: int = 30) -> None:
        """ログディレクトリと保持期間を受け取り初期化する。

        Args:
            log_dir: ログファイルを保存するディレクトリ。
            retention_days: ログを保持する日数。
        """
        self._log_dir = Path(log_dir)
        self._retention_days = retention_days
        self._configured = False

    def configure(self) -> None:
        """ログ設定を適用する。すでに設定済みなら何もしない。"""
        if self._configured:
            return

        # ログ出力ディレクトリが存在しない場合に備えて作成する
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # loguruのデフォルト設定を無効化し、アプリ専用の設定に置き換える
        logger.remove()
        log_file_pattern = self._log_dir / "run_{time:YYYYMMDD}.log"

        # 日次ローテーションと保持期間を設定したハンドラを追加する
        logger.add(
            log_file_pattern,
            rotation="00:00",
            retention=f"{self._retention_days} days",
            encoding="utf-8",
            enqueue=False,
            backtrace=True,
            diagnose=False,
        )

        self._configured = True

    def get_logger(self, name: Optional[str] = None) -> "Logger":
        """ログインスタンスを返す。"""
        self.configure()
        if name is None:
            return logger
        # 名前付きロガーを返し、用途に応じたコンテキスト情報を付与する
        return logger.bind(logger_name=name)
