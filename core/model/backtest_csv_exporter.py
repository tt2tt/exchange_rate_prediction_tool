"""バックテスト結果をCSVに出力するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .backtester import BacktestResult


@dataclass(frozen=True)
class ExportConfig:
    """CSV出力時の挙動を制御する設定値。"""

    output_dir: Path = Path("backtests")
    filename: str = "trades.csv"
    encoding: str = "utf-8"
    include_gross_return: bool = True


class BacktestCSVExporter:
    """バックテスト結果を所定の形式でCSV出力するクラス。"""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        """設定を受け取りエクスポーターを初期化する。

        Args:
            config: CSV出力に用いる設定。未指定時は既定値を利用する。
        """

        # 設定未指定時は標準設定を用いる
        self._config = config or ExportConfig()

    def export(self, result: BacktestResult, output_path: Optional[Path | str] = None) -> Path:
        """バックテスト結果をCSVに書き出す。

        Args:
            result: バックテストの結果オブジェクト。
            output_path: CSVの出力先。ファイルパスまたはディレクトリ。

        Returns:
            Path: 出力されたCSVファイルのパス。
        """

        # 出力先パスを決定し、必要ならディレクトリを作成する
        destination = self._resolve_output_path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # トレード単位へ集約したログを生成する
        summary = self._summarize_trades(result.trade_log)

        # CSVとして保存し、保存先パスを返す
        summary.to_csv(destination, index=False, encoding=self._config.encoding)
        return destination

    def _resolve_output_path(self, output_path: Optional[Path | str]) -> Path:
        """出力パスを組み立てる。

        Args:
            output_path: 呼び出し側から指定された出力先。

        Returns:
            Path: 実際に利用するファイルパス。
        """

        # 引数指定が無ければ既定のディレクトリ+ファイル名を利用する
        if output_path is None:
            return self._config.output_dir / self._config.filename
        candidate = Path(output_path)

        # ディレクトリが指定された場合は既定ファイル名を付与する
        if candidate.is_dir():
            return candidate / self._config.filename
        return candidate

    def _summarize_trades(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        """トレードログをエントリー単位に集約する。

        Args:
            trade_log: バックテスト結果のトレードログ。

        Returns:
            pd.DataFrame: エントリー・エグジット・損益情報をまとめた表。

        Raises:
            ValueError: 必須カラムが不足している場合。
        """

        # decision列が存在しない場合は処理できないため例外を投げる
        if "decision" not in trade_log.columns:
            raise ValueError("トレードログにdecision列が存在しません。")

        # 連続する同一decisionを1トレードとして識別するためのグルーピングIDを生成する
        decision_series = trade_log["decision"].fillna(0).astype(int)
        change_points = decision_series.ne(decision_series.shift(fill_value=0)).cumsum()

        records: list[dict[str, object]] = []

        for _, segment in trade_log.groupby(change_points):
            # decisionが0の場合はポジション無しなのでスキップする
            decision_value = int(segment["decision"].iloc[0])
            if decision_value == 0:
                continue

            # エントリー・エグジットの行インデックスを取得する
            entry_index = segment.index[0]
            exit_index = segment.index[-1]

            # 損益計算に必要なカラムが存在するか検証する
            net_return_series = self._safe_column(segment, "net_return")
            gross_return_series = self._safe_column(segment, "raw_return")

            if net_return_series is None:
                raise ValueError("トレードログにnet_return列が存在しません。")

            net_return = float(net_return_series.sum())

            record: dict[str, object] = {
                "entry_index": entry_index,
                "exit_index": exit_index,
                "direction": decision_value,
                "net_return": net_return,
                "duration": len(segment),
            }

            if self._config.include_gross_return and gross_return_series is not None:
                record["gross_return"] = float(gross_return_series.sum())

            records.append(record)

        # 1件もレコードが無い場合でも空のDataFrameを返す
        if not records:
            columns: list[str] = [
                "entry_index",
                "exit_index",
                "direction",
                "net_return",
                "duration",
            ]
            if self._config.include_gross_return:
                columns.append("gross_return")
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(records)

    @staticmethod
    def _safe_column(segment: pd.DataFrame, column: str) -> Optional[pd.Series]:
        """存在すれば対象カラムを返し、無ければNoneを返す。

        Args:
            segment: トレード区間のDataFrame。
            column: 取得対象のカラム名。

        Returns:
            Optional[pd.Series]: カラムが存在すればSeries、存在しない場合はNone。
        """

        # 欠落している場合に備えて存在確認を行う
        if column not in segment.columns:
            return None
        return segment[column]
