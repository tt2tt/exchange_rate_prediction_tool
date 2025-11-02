"""Backtesterの指標算出ロジックを検証するテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from core.model import BacktestConfig, Backtester


def _build_sample_trades() -> pd.DataFrame:
    """サンプルトレードのDataFrameを生成する。"""
    return pd.DataFrame(
        {
            "decision": [1, 0, 1, 1, 0],
            "actual_return": [0.02, 0.005, -0.01, 0.03, 0.0],
        }
    )


def test_run_returns_expected_metrics() -> None:
    """主要指標が期待通りに計算されることを確認する。"""
    trades = _build_sample_trades()
    config = BacktestConfig(spread=0.001, initial_capital=1.0)
    backtester = Backtester(config=config)

    result = backtester.run(trades)

    metrics = result.metrics

    assert metrics.total_return == pytest.approx(0.037, rel=1e-6)
    assert metrics.win_rate == pytest.approx(2 / 3, rel=1e-6)
    assert metrics.profit_factor == pytest.approx(0.048 / 0.011, rel=1e-6)
    assert metrics.max_drawdown > 0

    # エクイティカーブとトレードログが整合しているかを確認
    assert len(result.equity_curve) == len(trades)
    assert "cumulative_return" in result.trade_log.columns


def test_run_raises_when_no_trades() -> None:
    """トレードが存在しない場合に例外が発生することを確認する。"""
    trades = pd.DataFrame(
        {
            "decision": [0, 0, 0],
            "actual_return": [0.01, -0.02, 0.03],
        }
    )

    backtester = Backtester()

    with pytest.raises(ValueError):
        backtester.run(trades)
