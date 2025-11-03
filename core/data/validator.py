"""データセットの検証処理を提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

import re

import numpy as np
import pandas as pd

from .constants import COLUMN_ALIASES, PATTERN_COLUMN_ALIASES, REQUIRED_COLUMNS


@dataclass(frozen=True)
class ValidationResult:
    """検証結果を保持する。"""

    is_valid: bool
    messages: list[str]


class DataValidator:
    """CSVデータの検証を担当するクラス。"""

    def __init__(
        self,
        required_columns: Iterable[str] | None = None,
        column_aliases: Mapping[str, str] | None = None,
    ) -> None:
        """検証に必要な必須カラムや別名を設定する。"""
        self._required_columns = list(required_columns or REQUIRED_COLUMNS)
        default_aliases = dict(COLUMN_ALIASES)
        if column_aliases:
            default_aliases.update(column_aliases)
        self._column_aliases = default_aliases
        # 正規表現で括弧内の表記揺れを吸収するためのパターンを事前に準備する
        self._pattern_aliases: Tuple[tuple[re.Pattern[str], str], ...] = tuple(
            (re.compile(pattern), target) for pattern, target in PATTERN_COLUMN_ALIASES
        )

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """別名カラムを内部名称にリネームしたDataFrameを返す。"""
        # 取り込みデータは日本語名など複数のヘッダ形式が想定されるため
        # 事前に内部名称へ寄せておくことで後続処理をシンプルに保つ
        rename_map = {
            col: self._column_aliases[col]
            for col in df.columns
            if col in self._column_aliases
        }
        # 正規表現にマッチする列も内部名へリネームする
        for col in df.columns:
            if col in rename_map:
                continue
            for pattern, target in self._pattern_aliases:
                if pattern.match(col):
                    # 既に内部名が存在する場合は衝突を避けてスキップする
                    if target in df.columns:
                        break
                    rename_map[col] = target
                    break
        if not rename_map:
            return df.copy()
        normalized = df.copy()
        normalized = normalized.rename(columns=rename_map)
        return normalized

    def validate_columns(self, df: pd.DataFrame) -> ValidationResult:
        """必須カラムが揃っているかを検証する。"""
        available = set()

        for col in df.columns:
            if col in self._required_columns:
                available.add(col)
            elif col in self._column_aliases:
                available.add(self._column_aliases[col])
            else:
                for pattern, target in self._pattern_aliases:
                    if pattern.match(col):
                        available.add(target)
                        break

        missing = [col for col in self._required_columns if col not in available]
        if missing:
            message = f"必須カラムが不足しています: {', '.join(missing)}"
            return ValidationResult(False, [message])
        return ValidationResult(True, [])

    def validate_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """欠損値と無限値が含まれていないか検証する。"""
        messages: list[str] = []

        normalized_df = self.normalize_columns(df)

        # 欠損値の検出
        if normalized_df.isnull().any().any():
            messages.append("欠損値が含まれています。補完または除外してください。")

        # 無限大・無限小の検出
        numeric_df = normalized_df.select_dtypes(include=["number"])
        if not numeric_df.empty and np.isinf(numeric_df).any().any():
            messages.append("無限大/無限小の値が含まれています。")

        is_valid = not messages
        return ValidationResult(is_valid, messages)

    def run_all(self, df: pd.DataFrame) -> ValidationResult:
        """登録済みの検証をすべて実行する。"""
        messages: list[str] = []

        column_result = self.validate_columns(df)
        messages.extend(column_result.messages)

        if not column_result.is_valid:
            return ValidationResult(False, messages)

        missing_result = self.validate_missing_values(df)
        messages.extend(missing_result.messages)

        is_valid = not messages
        return ValidationResult(is_valid, messages)
