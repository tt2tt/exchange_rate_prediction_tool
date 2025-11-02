"""ロジスティック回帰モデルの学習処理を提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingConfig:
    """ロジスティック回帰の学習設定を保持する。"""

    penalty: str = "l2"
    c_value: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    class_weight: Optional[Mapping[int, float] | str] = None
    solver: str = "lbfgs"


@dataclass(frozen=True)
class TrainingResult:
    """学習済み推論器と付随情報をまとめるデータクラス。"""

    estimator: Pipeline
    accuracy: float
    label_mapping: dict[int, Any]
    messages: list[str]


class LogisticRegressionTrainer:
    """ロジスティック回帰モデルを学習させるクラス。"""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        """学習に用いる設定を初期化する。

        Args:
            config: 事前に用意した学習設定。未指定の場合は既定値を利用する。
        """

        # Configが渡されない場合は既定値を用いた設定を生成する
        self._config = config or TrainingConfig()

    def train(self, features: pd.DataFrame, target: pd.Series) -> TrainingResult:
        """ロジスティック回帰モデルを学習し結果を返す。

        Args:
            features: 学習に利用する特徴量。すべて数値である必要がある。
            target: 2値分類を想定した目的変数シリーズ。

        Returns:
            TrainingResult: 学習済み推論器と学習時のメトリクス。

        Raises:
            ValueError: データの検証に失敗した場合。
        """

        # 特徴量が空の場合は学習が成立しないためエラーとする
        if features.empty:
            raise ValueError("特徴量が空です。学習データを確認してください。")

        # ターゲット長と特徴量長が一致しないケースもエラーとする
        if len(features) != len(target):
            raise ValueError("特徴量と目的変数の件数が一致していません。")

        # ターゲットのユニーク数が1つの場合は分類が成立しない
        if target.nunique(dropna=False) < 2:
            raise ValueError("目的変数が単一クラスのため学習できません。")

        # 欠損値はLogisticRegressionが扱えないため事前検証する
        if features.isnull().any().any():
            raise ValueError("特徴量に欠損値が含まれています。前処理を確認してください。")

        if target.isnull().any():
            raise ValueError("目的変数に欠損値が含まれています。")

        # ラベルが非数値でも学習できるよう整数ラベルにエンコードする
        encoded_target, unique_labels = pd.factorize(target, sort=True)
        label_mapping = {int(index): label for index, label in enumerate(unique_labels)}
        y_values = encoded_target.astype(int)

        x_values = features.to_numpy(dtype=np.float64)

        # 標準化 + ロジスティック回帰のパイプラインを組み立てる
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        penalty=self._config.penalty,
                        C=self._config.c_value,
                        max_iter=self._config.max_iter,
                        random_state=self._config.random_state,
                        class_weight=self._config.class_weight,
                        solver=self._config.solver,
                    ),
                ),
            ]
        )

        # モデルを学習し、訓練データでの精度を計算する
        pipeline.fit(x_values, y_values)
        predictions = pipeline.predict(x_values)
        train_accuracy = float(accuracy_score(y_values, predictions))

        messages = ["ロジスティック回帰モデルの学習が完了しました。"]
        if label_mapping != {key: key for key in label_mapping}:
            messages.append(f"ラベル変換: {label_mapping}")
        messages.append(f"学習データに対する精度: {train_accuracy:.4f}")

        return TrainingResult(pipeline, train_accuracy, label_mapping, messages)
