"""学習済みモデルから予測結果を生成するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .logistic_regression_trainer import TrainingResult


@dataclass(frozen=True)
class PredictionResult:
    """推論結果と補足メッセージを保持するデータクラス。"""

    outputs: pd.DataFrame
    messages: list[str]


class Predictor:
    """ロジスティック回帰モデルの推論処理を提供するクラス。"""

    def __init__(
        self,
        training_result: TrainingResult,
        threshold: float = 0.5,
        positive_label_index: int = 1,
    ) -> None:
        """推論に必要な情報を初期化する。

        Args:
            training_result: 直近で学習した結果オブジェクト。
            threshold: 上昇と判定する確率の閾値。
            positive_label_index: ポジティブラベルを表すインデックス。
        """

        # 学習結果から推論器とラベル情報を保持する
        self._estimator = training_result.estimator
        self._label_mapping = training_result.label_mapping
        self._threshold = threshold
        self._positive_index = positive_label_index

        if positive_label_index not in self._label_mapping:
            raise ValueError("ポジティブラベルが学習結果に含まれていません。")

        self._positive_label: Any = self._label_mapping[positive_label_index]
        self._negative_label: Any = next(
            (label for index, label in self._label_mapping.items() if index != positive_label_index),
            None,
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """指定された特徴量に対して上昇確率と判定を算出する。

        Args:
            features: 推論対象となる特徴量データ。

        Returns:
            PredictionResult: 上昇確率と判定を含む推論結果を返す。
        """

        # 特徴量が空の場合は推論ができないためエラーを返す
        if features.empty:
            raise ValueError("推論用の特徴量が空です。")

        # 欠損値が残っている場合は学習時と整合しないためエラーとする
        if features.isnull().any().any():
            raise ValueError("推論用の特徴量に欠損値が含まれています。")

        # 推論器へ渡すために数値配列へ変換する
        x_values = features.to_numpy(dtype=np.float64)
        probabilities = self._estimator.predict_proba(x_values)

        if probabilities.shape[1] <= self._positive_index:
            raise ValueError("推論結果に期待するクラスが含まれていません。")

        positive_prob = probabilities[:, self._positive_index]

        # 予測ラベルは元のラベルへ復元して使いやすくする
        encoded_labels = self._estimator.predict(x_values).astype(int)
        decoded_labels = [self._label_mapping[int(label)] for label in encoded_labels]

        # 閾値を元にシグナル判定を行う
        decision_flags = (positive_prob >= self._threshold).astype(int)

        outputs = pd.DataFrame(index=features.index)
        outputs["probability_up"] = positive_prob
        outputs["predicted_label"] = decoded_labels
        outputs["decision"] = decision_flags
        if self._negative_label is not None:
            decision_labels = [self._positive_label if flag else self._negative_label for flag in decision_flags]
            outputs["decision_label"] = decision_labels

        messages = [
            f"{len(features)}件の予測結果を生成しました。",
            f"判定閾値: {self._threshold:.2f}",
        ]
        messages.append(f"ポジティブラベル: {self._positive_label}")
        if self._negative_label is not None:
            messages.append(f"ネガティブラベル: {self._negative_label}")

        return PredictionResult(outputs=outputs, messages=messages)
