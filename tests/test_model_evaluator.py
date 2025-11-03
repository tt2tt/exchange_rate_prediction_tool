"""ModelEvaluatorの挙動を検証するテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.model import (
    EvaluationConfig,
    ModelEvaluator,
    LogisticRegressionTrainer,
)


def _generate_time_series_dataset(rows: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """単純な時系列データセットを生成する。"""
    rng = np.random.default_rng(42)

    # 時系列を線形に増加させつつノイズを付与してシグナルを作成する
    timeline = np.arange(rows, dtype=float)
    base_signal = timeline / rows
    seasonal = np.sin(timeline / 3.0)
    noise = rng.normal(loc=0.0, scale=0.05, size=rows)

    feature_a = base_signal + seasonal + noise
    feature_b = base_signal - seasonal + noise

    features = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
        }
    )

    # シグナルが閾値を超えるかどうかで上昇(1)/下落(0)ラベルを決定する
    threshold = 0.5
    target = pd.Series((feature_a > threshold).astype(int), name="target")

    return features, target


def test_evaluate_returns_metrics_for_each_fold() -> None:
    """評価実行時に分割数とメトリクスが一致することを確認する。"""
    features, target = _generate_time_series_dataset()

    trainer = LogisticRegressionTrainer()
    evaluator = ModelEvaluator(trainer, EvaluationConfig(n_splits=4, test_size=5))

    result = evaluator.evaluate(features, target)

    assert len(result.fold_metrics) == 4
    assert set(result.overall_metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "expected_value",
    }

    for fold in result.fold_metrics:
        assert 0.0 <= fold.accuracy <= 1.0
        assert -1.0 <= fold.expected_value <= 1.0

    assert "TimeSeriesSplitによる評価が完了しました。" in result.messages[-1]


def test_evaluate_handles_small_gap_configuration() -> None:
    """gap付き分割でもメトリクスが算出できることを確認する。"""
    features, target = _generate_time_series_dataset(rows=30)

    trainer = LogisticRegressionTrainer()
    evaluator = ModelEvaluator(
        trainer,
        EvaluationConfig(n_splits=3, gap=1, test_size=5),
    )

    result = evaluator.evaluate(features, target)

    # 平均精度は有界値であることを確認する
