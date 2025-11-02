"""TimeSeriesSplitによるモデル評価を提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .logistic_regression_trainer import LogisticRegressionTrainer, TrainingResult


@dataclass(frozen=True)
class EvaluationConfig:
    """時系列交差検証の設定を保持する。"""

    n_splits: int = 5
    max_train_size: Optional[int] = None
    test_size: Optional[int] = None
    gap: int = 0


@dataclass(frozen=True)
class FoldMetrics:
    """各分割で算出された指標を格納するデータクラス。"""

    fold_index: int
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    expected_value: float


@dataclass(frozen=True)
class EvaluationResult:
    """時系列検証の結果をまとめて返す。"""

    fold_metrics: list[FoldMetrics]
    overall_metrics: dict[str, float]
    messages: list[str]


class ModelEvaluator:
    """ロジスティック回帰モデルの評価を担当するクラス。"""

    def __init__(
        self,
        trainer: LogisticRegressionTrainer,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        """評価器を初期化する。

        Args:
            trainer: 学習に利用する`LogisticRegressionTrainer`インスタンス。
            config: 分割数などを定義する評価設定。
        """

        # 依存オブジェクトと設定を保持して再利用する
        self._trainer = trainer
        self._config = config or EvaluationConfig()

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> EvaluationResult:
        """TimeSeriesSplitでロジスティック回帰を評価する。

        Args:
            features: モデル入力となる特徴量。
            target: 予測対象となる目的変数。

        Returns:
            EvaluationResult: 各分割の指標と全体平均を含む結果。
        """

        # 時系列分割の準備を行う
        splitter = TimeSeriesSplit(
            n_splits=self._config.n_splits,
            max_train_size=self._config.max_train_size,
            test_size=self._config.test_size,
            gap=self._config.gap,
        )

        fold_results: list[FoldMetrics] = []
        messages: list[str] = []

        # 期待値算出で利用する補助関数を定義する
        def _calc_expected_value(y_true: np.ndarray, proba: np.ndarray) -> float:
            # 上昇時は利益+1、下落時は-1とみなしたときの期待値を算出する
            gain_if_positive = proba
            gain_if_negative = -(1.0 - proba)
            expectation = np.where(y_true == 1, gain_if_positive, gain_if_negative)
            return float(np.mean(expectation))

        # 各分割で学習と評価を実施する
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(features), start=1):
            train_features = features.iloc[train_idx]
            train_target = target.iloc[train_idx]
            test_features = features.iloc[test_idx]
            test_target = target.iloc[test_idx]

            # 分割内でモデルを学習し、推論器とラベル対応を取得する
            training_result: TrainingResult = self._trainer.train(train_features, train_target)

            # 2クラス分類が前提のため、保有ラベル数をチェックする
            if len(training_result.label_mapping) != 2:
                raise ValueError("2クラス分類以外は評価対象外です。")

            estimator = training_result.estimator
            label_to_index = {label: index for index, label in training_result.label_mapping.items()}

            # テストデータのラベルを学習時のインデックスへ変換する
            mapped_target = test_target.map(label_to_index)
            if mapped_target.isna().any():
                raise ValueError("学習時に存在しないラベルが含まれています。")
            encoded_test_target = mapped_target.to_numpy(dtype=int)

            x_test = test_features.to_numpy(dtype=np.float64)

            # テストデータに対する予測と確率を算出する
            predicted_labels = estimator.predict(x_test)
            probabilities = estimator.predict_proba(x_test)
            positive_class_index = 1
            positive_prob = probabilities[:, positive_class_index]

            # ROC-AUCは正例と負例が両方存在する場合のみ計算する
            if np.unique(encoded_test_target).size < 2:
                roc_auc = float("nan")
                messages.append(f"Fold{fold_index}: ROC-AUCは計算できませんでした。")
            else:
                roc_auc = float(roc_auc_score(encoded_test_target, positive_prob))

            accuracy = float(accuracy_score(encoded_test_target, predicted_labels))
            precision = float(precision_score(encoded_test_target, predicted_labels, zero_division=0))
            recall = float(recall_score(encoded_test_target, predicted_labels, zero_division=0))
            expected_value = _calc_expected_value(encoded_test_target, positive_prob)

            fold_metric = FoldMetrics(
                fold_index=fold_index,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                roc_auc=roc_auc,
                expected_value=expected_value,
            )
            fold_results.append(fold_metric)

            messages.append(
                "Fold{index}: accuracy={acc:.3f}, precision={prec:.3f}, "
                "recall={rec:.3f}, roc_auc={auc:.3f}, expected={exp:.3f}".format(
                    index=fold_index,
                    acc=accuracy,
                    prec=precision,
                    rec=recall,
                    auc=roc_auc if not np.isnan(roc_auc) else float("nan"),
                    exp=expected_value,
                )
            )

        # 平均値を集計し全体指標としてまとめる
        def _mean(values: list[float]) -> float:
            array = np.array(values, dtype=float)
            return float(np.nanmean(array))

        overall_metrics = {
            "accuracy": _mean([fold.accuracy for fold in fold_results]),
            "precision": _mean([fold.precision for fold in fold_results]),
            "recall": _mean([fold.recall for fold in fold_results]),
            "roc_auc": _mean([fold.roc_auc for fold in fold_results]),
            "expected_value": _mean([fold.expected_value for fold in fold_results]),
        }

        messages.append("TimeSeriesSplitによる評価が完了しました。")

        return EvaluationResult(fold_results, overall_metrics, messages)
