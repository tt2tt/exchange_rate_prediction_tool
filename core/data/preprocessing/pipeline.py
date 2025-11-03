"""データ前処理パイプラインの実装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from core.data.validator import DataValidator
from core.data.constants import REQUIRED_COLUMNS


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

        coerced_df, coercion_messages = self._coerce_numeric_columns(normalized)
        messages.extend(coercion_messages)

        sorted_df = self._sort_by_date(coerced_df)
        messages.append("Date列を昇順ソートしました。")

        return PreprocessingResult(sorted_df, messages)

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
        """数値列を強制的にfloatへ変換し、文字列混入を排除する。"""
        # Dateを除いた必須カラムを対象に、文字列やハイフンが混在してもNaNへ変換する
        numeric_columns = [col for col in REQUIRED_COLUMNS if col != "Date" and col in df.columns]
        messages: list[str] = []
        coerced = df.copy()

        for column in numeric_columns:
            # 変換前と後でNaN数を比較し、今回の変換で失われた値を計測する
            before_nan = coerced[column].isna().sum()
            numeric_series = pd.to_numeric(coerced[column], errors="coerce")
            after_nan = numeric_series.isna().sum()

            if after_nan > before_nan:
                converted_count = after_nan - before_nan
                messages.append(f"{column}列で数値以外の値を{converted_count}件検出しNaNに変換しました。")

            coerced[column] = numeric_series

        return coerced, messages