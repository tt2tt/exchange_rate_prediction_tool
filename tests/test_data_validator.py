"""DataValidatorの確認テスト。"""

from typing import cast

import numpy as np
import pandas as pd

from core.data.validator import DataValidator


def test_validate_columns_detects_missing_columns() -> None:
    """必須カラム不足を検出できること。"""
    df = pd.DataFrame({"Open": [1], "Close": [1]})
    validator = DataValidator()

    result = validator.validate_columns(df)

    assert not result.is_valid
    assert "必須カラム" in result.messages[0]


def test_validate_missing_values_detects_nan_and_inf() -> None:
    """欠損値および無限値を検出できること。"""
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Open": [1.0],
        "High": [float("inf")],
        "Low": [0.5],
        "Close": [float("nan")],
        "MA5": [1.0],
        "MA25": [1.0],
        "MA75": [1.0],
    })
    validator = DataValidator()

    result = validator.validate_missing_values(df)

    assert not result.is_valid
    assert any("欠損値" in message for message in result.messages)
    assert any("無限" in message for message in result.messages)


def test_run_all_returns_valid_when_clean() -> None:
    """クリーンなデータでは検証が成功すること。"""
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Open": [1.0],
        "High": [1.2],
        "Low": [0.9],
        "Close": [1.1],
        "MA5": [1.05],
        "MA25": [1.03],
        "MA75": [1.01],
    })
    validator = DataValidator()

    result = validator.run_all(df)

    assert result.is_valid
    assert not result.messages


def test_run_all_accepts_japanese_headers() -> None:
    """日本語カラムでも検証を通過すること。"""
    df = pd.DataFrame({
        "日付": ["2025/10/17"],
        "始値": [150.359],
        "高値": [150.593],
        "安値": [149.371],
        "終値": [150.582],
        "期間A[5](日足)": [151.237],
        "期間C[25](日足)": [149.382],
        "期間G[75](日足)": [148.205],
    })
    validator = DataValidator()

    result = validator.run_all(df)

    assert result.is_valid
    assert not result.messages


def test_normalize_columns_returns_canonical_names() -> None:
    """別名カラムが内部名にリネームされること。"""
    df = pd.DataFrame({
        "日付": ["2025/10/17"],
        "始値": [150.0],
        "終値": [151.0],
        "期間A[5](日足)": [150.5],
    })
    validator = DataValidator()

    normalized = validator.normalize_columns(df)

    assert set(normalized.columns) == {"Date", "Open", "Close", "MA5"}
    ma5_value = cast(float, normalized.loc[0, "MA5"])
    assert np.isclose(ma5_value, 150.5)
