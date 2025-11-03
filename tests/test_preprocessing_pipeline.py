"""PreprocessingPipelineのテスト。"""

from __future__ import annotations

from typing import cast

import pandas as pd

from core.data.preprocessing.pipeline import PreprocessingPipeline


def test_preprocess_sorts_by_date() -> None:
    """Date列を昇順にソートする。"""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-01"],
            "Open": [2.0, 1.0],
            "High": [2.5, 1.5],
            "Low": [1.8, 0.8],
            "Close": [2.1, 1.1],
            "MA5": [2.0, 1.0],
            "MA25": [2.0, 1.0],
            "MA75": [2.0, 1.0],
        }
    )

    pipeline = PreprocessingPipeline()
    result = pipeline.preprocess(df)

    date_value = cast("pd.Timestamp", result.dataframe.loc[0, "Date"])
    assert date_value.strftime("%Y-%m-%d") == "2024-01-01"
    assert "Date列を昇順ソートしました。" in result.messages


def test_preprocess_returns_messages_on_validation_failure() -> None:
    """検証失敗時は元データとエラーメッセージを返す。"""
    df = pd.DataFrame({"Open": [1.0]})

    pipeline = PreprocessingPipeline()
    result = pipeline.preprocess(df)

    assert result.dataframe.equals(df.reset_index(drop=True))
    assert any("必須カラム" in message for message in result.messages)
