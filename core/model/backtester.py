"""シンプルなバックテスト機能を提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """バックテストに使用する設定値を保持する。"""

    spread: float = 0.0005
    initial_capital: float = 1.0


@dataclass(frozen=True)
class BacktestMetrics:
    """バックテストで算出された指標をまとめるデータクラス。"""

    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    gross_profit: float
    gross_loss: float


@dataclass(frozen=True)
class BacktestResult:
    """バックテストの結果を包括的に保持する。"""

    metrics: BacktestMetrics
    trade_log: pd.DataFrame
    equity_curve: pd.Series
    messages: list[str]


class Backtester:
    """ロジック化されたバックテストを実行するクラス。"""

    REQUIRED_COLUMNS = ("decision", "actual_return")

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """バックテスターを初期化する。

        Args:
            config: スプレッドや初期資金を指定する設定オブジェクト。
        """

        # 設定が与えられない場合は既定値を用いる
        self._config = config or BacktestConfig()

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """バックテストを実行し結果を返す。

        Args:
            df: `decision`列と`actual_return`列を含むDataFrame。

        Returns:
            BacktestResult: 指標・トレードログ・エクイティカーブを含む結果。

        Raises:
            ValueError: 入力検証に失敗した場合。
        """

        # 必須カラムが揃っているかを確認する
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {', '.join(missing_columns)}")

        # 欠損値のままでは指標が破綻するため事前に弾く
        if df[list(self.REQUIRED_COLUMNS)].isnull().any().any():
            raise ValueError("decisionまたはactual_returnに欠損値が含まれています。")

        # トレード判定が存在しない場合はバックテストが成立しない
        trade_mask = df["decision"] != 0
        trade_count = int(trade_mask.sum())
        if trade_count == 0:
            raise ValueError("トレード判定が存在しないためバックテストを実行できません。")

        # スプレッド適用後の純粋なリターンを算出する
        raw_returns = df["actual_return"].to_numpy(dtype=np.float64)
        decisions = df["decision"].to_numpy(dtype=np.int64)
        net_returns = np.where(decisions != 0, raw_returns - self._config.spread, 0.0)

        # トレードごとのリターンをデータフレーム化して閲覧しやすくする
        trade_log = pd.DataFrame(index=df.index)
        trade_log["decision"] = decisions
        trade_log["raw_return"] = raw_returns
        trade_log["net_return"] = net_returns

        # 勝率やPF算出のために正負別で集計する
        trade_returns = net_returns[trade_mask.to_numpy()]
        wins = np.sum(trade_returns > 0)
        gross_profit = float(np.sum(trade_returns[trade_returns > 0]))
        gross_loss = float(-np.sum(trade_returns[trade_returns < 0]))

        win_rate = float(wins / trade_count)
        if gross_loss == 0.0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = float(gross_profit / gross_loss)

        # 累積損益から最大ドローダウンを計算する
        equity_curve = self._compute_equity_curve(net_returns, df.index)
        max_drawdown = self._compute_max_drawdown(equity_curve)

        total_return = float(np.sum(trade_returns))

        metrics = BacktestMetrics(
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
        )

        # トレード毎の累積損益を記録する
        trade_log["cumulative_return"] = equity_curve - self._config.initial_capital

        messages = [
            f"トレード数: {trade_count}",
            f"勝率: {win_rate:.3f}",
            f"プロフィットファクター: {profit_factor if np.isfinite(profit_factor) else 'inf'}",
            f"最大ドローダウン: {max_drawdown:.4f}",
            f"総リターン: {total_return:.4f}",
        ]

        return BacktestResult(
            metrics=metrics,
            trade_log=trade_log,
            equity_curve=equity_curve,
            messages=messages,
        )

    def _compute_equity_curve(self, net_returns: np.ndarray, index: pd.Index) -> pd.Series:
        """純リターンからエクイティカーブを算出する。

        Args:
            net_returns: スプレッド適用後の各期間リターン。
            index: 元のDataFrameのインデックス。

        Returns:
            pd.Series: 各期間の資産推移を表すエクイティカーブ。
        """

        # （1 + リターン）を累積積で掛け合わせることで資産推移を得る
        growth_factors = 1.0 + net_returns
        equity = self._config.initial_capital * np.cumprod(growth_factors)
        return pd.Series(equity, index=index)

    def _compute_max_drawdown(self, equity_curve: pd.Series) -> float:
        """エクイティカーブから最大ドローダウンを計測する。

        Args:
            equity_curve: 資産推移を表すエクイティカーブ。

        Returns:
            float: 最大ドローダウンの大きさ（正の値）。
        """

        # 過去最高値との差分を割合で計算し、最も落ち込んだ値を採用する
        cumulative_max = equity_curve.cummax()
        drawdowns = (equity_curve - cumulative_max) / cumulative_max
        return float(abs(drawdowns.min()))
