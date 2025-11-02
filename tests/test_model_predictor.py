"""Predictorクラスの挙動を検証するテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from core.model import LogisticRegressionTrainer, Predictor


def _build_linear_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """単純な線形分離可能データセットを作成する。"""
    features = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 1.0, 1.2, 1.4],
            "feature_b": [1.0, 0.9, 1.1, 2.0, 2.2, 2.4],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1], name="target")
    return features, target


def test_predictor_returns_probability_and_decision() -> None:
    """推論結果に確率と判定が含まれることを確認する。"""
    features, target = _build_linear_dataset()

    # 事前にロジスティック回帰を学習して推論器を用意する
    training_result = LogisticRegressionTrainer().train(features, target)
    predictor = Predictor(training_result=training_result, threshold=0.6)

    result = predictor.predict(features)

    assert list(result.outputs.columns) == [
        "probability_up",
        "predicted_label",
        "decision",
        "decision_label",
    ]
    assert result.outputs["probability_up"].between(0.0, 1.0).all()
    assert set(result.outputs["decision"]) <= {0, 1}
    assert set(result.outputs["decision_label"]) <= {0, 1}
    assert any(msg.startswith("ポジティブラベル") for msg in result.messages)


def test_predictor_raises_on_empty_features() -> None:
    """特徴量が空の場合に例外が送出されることを確認する。"""
    features, target = _build_linear_dataset()
    training_result = LogisticRegressionTrainer().train(features, target)
    predictor = Predictor(training_result=training_result)

    empty_features = pd.DataFrame(columns=features.columns)

    with pytest.raises(ValueError):
        predictor.predict(empty_features)