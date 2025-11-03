"""特徴量生成を担当するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.data.validator import DataValidator


@dataclass
class FeatureResult:
    """生成された特徴量とメタ情報を保持するデータクラス。"""

    features: pd.DataFrame
    messages: list[str]


class FeatureEngineer:
    """学習用特徴量を生成するクラス。"""

    def __init__(self, validator: Optional[DataValidator] = None) -> None:
        """依存するバリデータを設定する。

        Args:
            validator: 外部から注入する`DataValidator`。未指定時は自前で生成する。
        """

        self._validator = validator or DataValidator()

    def _safe_ratio(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """ゼロ除算を避けつつ比率を計算する。

        Args:
            numerator: 分子となる数列。
            denominator: 分母となる数列。

        Returns:
            pd.Series: ゼロ除算を0で埋めた比率値。
        """

        denom = denominator.replace(to_replace=0, value=np.nan)
        ratio = numerator / denom
        return ratio.fillna(0.0)

    def _calc_cross_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """移動平均線のゴールデン/デッドクロスを検出する。

        Args:
            df: 移動平均の値を含むデータフレーム。

        Returns:
            pd.DataFrame: クロスイベントを表すOne-Hot特徴量。
        """

        ma5_above = df["MA5"] > df["MA25"]
        prev_ma5_above = ma5_above.shift(1, fill_value=False)

        golden_cross = (ma5_above & ~prev_ma5_above).astype(int)
        dead_cross = (~ma5_above & prev_ma5_above).astype(int)

        return pd.DataFrame(
            {
                "feature_golden_cross": golden_cross,
                "feature_dead_cross": dead_cross,
            }
        )

    def generate(self, df: pd.DataFrame, lag: int) -> FeatureResult:
        """指定したラグを適用した特徴量を返す。

        Args:
            df: 入力となる終値や移動平均を含むデータフレーム。
            lag: 何行分ずらして特徴量を利用するかのラグ値。

        Returns:
            FeatureResult: 生成された特徴量と補足メッセージ。
        """

        messages: list[str] = []

        column_result = self._validator.validate_columns(df)
        if not column_result.is_valid:
            messages.extend(column_result.messages)
            return FeatureResult(pd.DataFrame(index=df.index), messages)

        normalized = self._validator.normalize_columns(df)

        missing_result = self._validator.validate_missing_values(normalized)
        messages.extend(missing_result.messages)

        features = pd.DataFrame(index=normalized.index)

        # 主要カラムを取得しておくと冗長な辞書参照を減らせる
        close = normalized["Close"]
        open_ = normalized["Open"]
        high = normalized["High"]
        low = normalized["Low"]
        ma5 = normalized["MA5"]
        ma25 = normalized["MA25"]
        ma75 = normalized["MA75"]

        # リターン・ローソク足の形状に関する指標
        ret = close.pct_change()
        features["feature_ret1"] = ret

        body = close - open_
        range_ = high - low
        features["feature_body_ratio"] = self._safe_ratio(body, range_)
        features["feature_range_to_close"] = self._safe_ratio(range_, close)

        # 移動平均線との乖離率を算出する
        features["feature_ma5_deviation"] = self._safe_ratio(close, ma5) - 1.0
        features["feature_ma25_deviation"] = self._safe_ratio(close, ma25) - 1.0
        features["feature_ma75_deviation"] = self._safe_ratio(close, ma75) - 1.0

        # 移動平均線の傾きを正規化した斜率で表現する
        features["feature_ma5_slope"] = self._safe_ratio(ma5.diff(), ma5)
        features["feature_ma25_slope"] = self._safe_ratio(ma25.diff(), ma25)
        features["feature_ma75_slope"] = self._safe_ratio(ma75.diff(), ma75)

        cross_flags = self._calc_cross_flags(normalized)
        features = pd.concat([features, cross_flags], axis=1)

        # 短期・中期のボラティリティを取得する
        vol_window_24 = ret.rolling(24, min_periods=1).std()
        vol_window_72 = ret.rolling(72, min_periods=1).std()
        features["feature_volatility_24"] = vol_window_24
        features["feature_volatility_72"] = vol_window_72

        if lag > 0:
            features = features.shift(lag)
            messages.append(f"特徴量にラグ{lag}を適用しました。")
        else:
            messages.append("ラグは0として特徴量を使用します。")

        features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return FeatureResult(features, messages)
