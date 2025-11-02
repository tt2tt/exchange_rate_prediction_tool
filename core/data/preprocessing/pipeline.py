"""データ前処理パイプラインの実装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.data.validator import DataValidator


@dataclass
class PreprocessingResult:
    """前処理の結果とログを保持する。"""

    dataframe: pd.DataFrame
    messages: list[str]


class PreprocessingPipeline:
    """CSVデータに対する前処理手順をまとめたクラス。"""

    def __init__(self, validator: Optional[DataValidator] = None) -> None:
        self._validator = validator or DataValidator()

    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Date列が存在する場合は昇順にソートする。"""
        if "Date" in df.columns:
            # Date列が文字列の場合でもソートできるように変換を行う
            sorted_df = df.copy()
            sorted_df["Date"] = pd.to_datetime(sorted_df["Date"], errors="coerce")
            sorted_df = sorted_df.sort_values("Date", ascending=True)
            return sorted_df.reset_index(drop=True)
        # Date列が無い場合はそのまま返す
        return df.reset_index(drop=True)

    def preprocess(self, df: pd.DataFrame) -> PreprocessingResult:
        """検証とソートを含む基本的な前処理を適用する。"""
        messages: list[str] = []

        validation = self._validator.run_all(df)
        if not validation.is_valid:
            messages.extend(validation.messages)
            return PreprocessingResult(df.reset_index(drop=True), messages)

        normalized = self._validator.normalize_columns(df)
        sorted_df = self._sort_by_date(normalized)
        messages.append("Date列を昇順ソートしました。")

        return PreprocessingResult(sorted_df, messages)