"""LogisticRegressionTrainerの挙動を検証するテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from core.model import LogisticRegressionTrainer, TrainingConfig


def _build_linear_separable_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """線形分離可能なサンプルデータを生成する。"""
    # シンプルな特徴量を構築してクラス間の境界を明確にする
    features = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 1.0, 1.2, 1.4],
            "feature_b": [1.0, 0.8, 1.2, 2.0, 2.2, 2.4],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1], name="target")
    return features, target


def test_train_returns_trained_pipeline() -> None:
    """学習完了後に精度とメッセージが返ることを確認する。"""
    features, target = _build_linear_separable_dataset()

    # 既定設定でトレーナーを生成し、学習を実施する
    trainer = LogisticRegressionTrainer()
    result = trainer.train(features, target)

    lr_model = result.estimator.named_steps["logistic_regression"]

    # パイプラインには学習済み係数が含まれ、精度が高いことを検証する
    assert lr_model.coef_.shape[0] == 1
    assert result.accuracy == pytest.approx(1.0)
    assert result.label_mapping == {0: 0, 1: 1}
    assert "ロジスティック回帰モデルの学習が完了しました。" in result.messages[0]


def test_train_respects_custom_config() -> None:
    """カスタム設定がロジスティック回帰へ反映されることを確認する。"""
    features, target = _build_linear_separable_dataset()

    # 高速収束のために反復回数を制限した設定を利用する
    config = TrainingConfig(max_iter=200, c_value=0.5, solver="lbfgs")
    trainer = LogisticRegressionTrainer(config=config)
    result = trainer.train(features, target)

    lr_model = result.estimator.named_steps["logistic_regression"]

    # モデルに設定が反映されているかを確認する
    assert lr_model.max_iter == 200
    assert lr_model.C == pytest.approx(0.5)


def test_train_raises_when_target_is_single_class() -> None:
    """ターゲットが単一クラスの場合に例外が発生することを確認する。"""
    features, _ = _build_linear_separable_dataset()
    target = pd.Series([1, 1, 1, 1, 1, 1], name="target")

    trainer = LogisticRegressionTrainer()

    with pytest.raises(ValueError):
        trainer.train(features, target)
