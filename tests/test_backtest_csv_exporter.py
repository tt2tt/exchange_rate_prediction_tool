"""BacktestCSVExporterの挙動を検証するテスト。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.model import BacktestCSVExporter, BacktestConfig, Backtester, ExportConfig


def _build_sample_trades() -> pd.DataFrame:
    """サンプルトレードのDataFrameを生成する。"""

    return pd.DataFrame(
        {
            "decision": [1, 0, 1, 1, 0, -1, -1, 0],
            "actual_return": [0.02, 0.0, -0.01, 0.03, 0.0, -0.015, 0.005, 0.0],
        }
    )


def test_export_creates_csv_with_trade_summary(tmp_path: Path) -> None:
    """CSVが生成され内容が期待通りであることを確認する。"""

    trades = _build_sample_trades()
    backtester = Backtester(config=BacktestConfig(spread=0.001))
    result = backtester.run(trades)

    exporter = BacktestCSVExporter()
    output_path = exporter.export(result, tmp_path / "reports" / "trades.csv")

    assert output_path.exists()

    exported = pd.read_csv(output_path)

    assert {
        "entry_index",
        "exit_index",
        "direction",
        "net_return",
        "duration",
        "gross_return",
    } <= set(exported.columns)

    # 3つのトレード（ロング2回・ショート1回）が記録されていることを確認する
    assert len(exported) == 3

    # それぞれの方向が期待通りか検証する
    assert exported["direction"].tolist() == [1, 1, -1]


def test_export_uses_default_path(tmp_path: Path) -> None:
    """出力先未指定の場合に既定パスが利用されることを検証する。"""

    trades = _build_sample_trades()
    backtester = Backtester()
    result = backtester.run(trades)

    exporter = BacktestCSVExporter(
        config=ExportConfig(output_dir=tmp_path, filename="summary.csv")
    )

    csv_path = exporter.export(result)

    assert csv_path == tmp_path / "summary.csv"
    assert csv_path.exists()


def test_export_raises_when_decision_missing(tmp_path: Path) -> None:
    """decision列欠如時に例外が発生することを確認する。"""

    exporter = BacktestCSVExporter()

    trade_log = pd.DataFrame({"actual_return": [0.01, -0.02]})

    with pytest.raises(ValueError):
        exporter._summarize_trades(trade_log)  # type: ignore[attr-defined]
