"""FeatureEngineerの挙動を検証するテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from core.features import FeatureEngineer


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """テスト用の基本データセットを生成する。"""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")

    # 移動平均線が交差するように値を調整する
    close_prices = [100.0, 101.0, 102.5, 101.5, 103.0]
    open_prices = [99.5, 100.5, 102.0, 101.0, 102.5]
    high_prices = [100.5, 101.5, 103.0, 102.0, 103.5]
    low_prices = [99.0, 100.0, 101.5, 100.5, 102.0]
    ma5 = [99.5, 100.5, 101.5, 101.8, 102.2]
    ma25 = [100.0, 100.2, 100.4, 100.6, 100.8]
    ma75 = [100.0, 100.1, 100.2, 100.3, 100.4]

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "MA5": ma5,
            "MA25": ma25,
            "MA75": ma75,
        }
    )


def test_generate_returns_expected_columns(sample_dataframe: pd.DataFrame) -> None:
    """生成された特徴量のカラム構成と欠損が無いことを確認する。"""
    engineer = FeatureEngineer()

    # ラグ無しで特徴量を生成する
    result = engineer.generate(sample_dataframe, lag=0)

    expected_columns = [
        "feature_ret1",
        "feature_body_ratio",
        "feature_range_to_close",
        "feature_ma5_deviation",
        "feature_ma25_deviation",
        "feature_ma75_deviation",
        "feature_ma5_slope",
        "feature_ma25_slope",
        "feature_ma75_slope",
        "feature_golden_cross",
        "feature_dead_cross",
        "feature_volatility_24",
        "feature_volatility_72",
    ]

    assert list(result.features.columns) == expected_columns
    assert result.features.isna().sum().sum() == 0
    assert "ラグは0として特徴量を使用します。" in result.messages


def test_generate_applies_lag(sample_dataframe: pd.DataFrame) -> None:
    """ラグ指定で先頭行がシフトされることを検証する。"""
    engineer = FeatureEngineer()

    # ラグを1に設定して遅延させる
    result = engineer.generate(sample_dataframe, lag=1)

    # pct_changeの2行目とラグ済み先頭行が一致することを期待する
    expected_first_ret = result.features.loc[1, "feature_ret1"]
    assert result.features.loc[0, "feature_ret1"] == pytest.approx(expected_first_ret)
    assert "特徴量にラグ1を適用しました。" in result.messages


def test_generate_handles_invalid_dataframe() -> None:
    """必須カラム欠如時に空データとメッセージが返ることを確認する。"""
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "MA5": [100.0, 100.5, 101.0],
            "MA25": [100.0, 100.1, 100.2],
            # MA75列を意図的に欠落させる
        }
    )

    result = FeatureEngineer().generate(df, lag=0)

    assert result.features.empty
    assert "必須カラムが不足しています" in "".join(result.messages)
