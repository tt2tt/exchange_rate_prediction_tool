"""モデル保存とログ出力を司るモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib  # type: ignore[import-untyped]


@dataclass(frozen=True)
class ModelPersistenceResult:
    """モデル保存と併せて生成された成果物のパスをまとめる。"""

    model_path: Path
    log_path: Path


class ModelPersistenceManager:
    """学習済みモデルの保存とログ出力を担当するクラス。"""

    def __init__(
        self,
        model_dir: Path | str = Path("models"),
        log_dir: Path | str = Path("logs"),
        retention_days: int = 30,
    ) -> None:
        """保存に使用するディレクトリと保持期間を初期化する。"""

        # ルートディレクトリを Path 化し、後続処理で使いやすくしておく
        self._model_dir = Path(model_dir)
        self._log_dir = Path(log_dir)
        self._retention_days = retention_days

    def save(
        self,
        estimator: Any,
        messages: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelPersistenceResult:
        """学習済みモデルを保存し、実行ログを出力する。"""

        # 保存先ディレクトリを事前に用意し、存在しない場合は作成する
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # モデルは常に latest_model.joblib として上書き保存する
        model_path = self._model_dir / "latest_model.joblib"
        joblib.dump(estimator, model_path)

        # タイムスタンプ付きのログファイル名を生成し、人が追跡しやすくする
        timestamp = datetime.now()
        log_filename = f"run_{timestamp.strftime('%Y%m%d_%H%M')}.txt"
        log_path = self._log_dir / log_filename

        # メタデータとメッセージを整形してログに書き込む
        lines: list[str] = [f"timestamp={timestamp.isoformat(timespec='seconds')}" ]
        if metadata:
            for key, value in metadata.items():
                lines.append(f"{key}={value}")
        if messages:
            lines.append("messages=")
            lines.extend(messages)

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("\n".join(lines))

        # 古いログを保持期間外に出ないよう整理する
        self._cleanup_old_logs()

        return ModelPersistenceResult(model_path=model_path, log_path=log_path)

    def _cleanup_old_logs(self) -> None:
        """保持期間を過ぎたログファイルを削除する。"""

        # 削除対象の基準日時を計算し、その日時より古いファイルを除去する
        cutoff = datetime.now() - timedelta(days=self._retention_days)
        for log_file in self._log_dir.glob("run_*.txt"):
            stem = log_file.stem
            try:
                timestamp_str = stem.split("_", maxsplit=1)[1]
                log_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
            except (IndexError, ValueError):
                continue
            if log_time < cutoff:
                log_file.unlink(missing_ok=True)
