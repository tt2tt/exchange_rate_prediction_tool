"""UIメインウィンドウのモックを検証するテスト。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pytestqt.qtbot import QtBot  # type: ignore[import-not-found]
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QPushButton
from sklearn.pipeline import Pipeline

from core.model.backtester import BacktestMetrics, BacktestResult
from core.model.evaluation import EvaluationResult
from core.model.logistic_regression_trainer import TrainingResult
from core.model.predictor import PredictionResult
from ui.main_window import ExecutionSummary, MainWindow


def _create_window(qtbot: QtBot) -> MainWindow:
    """MainWindowインスタンスを生成してqtbotへ登録する。"""
    # ウィンドウを生成し、テスト後に後始末されるようqtbotへ登録する
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    main_window.show()
    return main_window


def test_main_window_has_core_widgets(qtbot: QtBot) -> None:
    """主要ウィジェットが配置されていることを確認する。"""
    window = _create_window(qtbot)
    # ファイルパス入力欄の存在を検証する
    file_input = window.findChild(QLineEdit, "filePathInput")
    assert file_input is not None

    # ファイル選択ボタンが表示されているか確認する
    file_button = window.findChild(QPushButton, "fileSelectButton")
    assert file_button is not None

    # 実行ボタンが存在し有効になっていることを確認する
    execute_button = window.findChild(QPushButton, "executeButton")
    assert execute_button is not None
    assert execute_button.isEnabled()

    # 進捗・ログ・結果ラベルの初期状態を確認する
    assert window.progress_bar.value() == 0
    assert window.log_output.toPlainText() == ""
    assert "未実行" in window.evaluation_label.text()
    assert "未実行" in window.training_label.text()
    assert "未実行" in window.prediction_label.text()
    assert "未実行" in window.backtest_label.text()


def test_spread_input_has_default_value(qtbot: QtBot) -> None:
    """スプレッド入力欄の初期値と設定を検証する。"""
    window = _create_window(qtbot)
    spread_input = window.findChild(QDoubleSpinBox, "spreadInput")
    assert spread_input is not None

    # 初期値と刻み幅が想定どおりであることを確認する
    assert spread_input.value() == pytest.approx(0.0005)
    assert spread_input.singleStep() == pytest.approx(0.0001)
    assert spread_input.maximum() == pytest.approx(0.01)


def test_validate_dataframe_detects_missing_columns(qtbot: QtBot) -> None:
    """必須カラム不足時にエラーメッセージが返ることを確認する。"""
    window = _create_window(qtbot)
    dataframe = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
        "Open": range(12),
        "High": range(12),
        "Low": range(12),
        "Close": range(12),
    })

    errors = window._validate_dataframe(dataframe)

    assert any("必須カラム" in message for message in errors)


def test_validate_dataframe_requires_min_rows(qtbot: QtBot) -> None:
    """行数不足時にエラーが返ることを確認する。"""
    window = _create_window(qtbot)
    columns = {
        "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Open": range(5),
        "High": range(5),
        "Low": range(5),
        "Close": range(5),
        "MA5": range(5),
        "MA25": range(5),
        "MA75": range(5),
    }
    dataframe = pd.DataFrame(columns)

    errors = window._validate_dataframe(dataframe)

    assert any("データ件数" in message for message in errors)


def test_on_execute_shows_error_when_file_missing(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """ファイル未選択時にエラーダイアログが呼び出されることを確認する。"""
    window = _create_window(qtbot)

    captured: dict[str, str | None] = {"error": None}

    def fake_critical(parent, title, message):  # type: ignore[override]
        captured["error"] = message

    monkeypatch.setattr("ui.main_window.QMessageBox.critical", fake_critical)

    window._on_execute()

    assert captured["error"] is not None


def test_on_execute_shows_info_when_validation_succeeds(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """検証成功時に情報ダイアログが呼び出されることを確認する。"""

    window = _create_window(qtbot)

    dataframe = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "Open": range(12),
            "High": range(12),
            "Low": range(12),
            "Close": range(12),
            "MA5": range(12),
            "MA25": range(12),
            "MA75": range(12),
        }
    )

    monkeypatch.setattr(window, "_read_csv", lambda _: dataframe)
    captured: dict[str, str | None] = {"info": None}

    evaluation_result = EvaluationResult(
        fold_metrics=[],
        overall_metrics={
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "roc_auc": 0.6,
            "expected_value": 0.05,
        },
        messages=["評価ログ"],
    )
    training_result = TrainingResult(
        estimator=Pipeline([]),
        accuracy=0.92,
        label_mapping={0: 0, 1: 1},
        messages=["学習ログ"],
        model_path=tmp_path / "latest_model.joblib",
        log_path=tmp_path / "run.log",
    )
    prediction_outputs = pd.DataFrame(
        {
            "probability_up": [0.6, 0.4],
            "predicted_label": [1, 0],
            "decision": [1, 0],
        }
    )
    prediction_result = PredictionResult(outputs=prediction_outputs, messages=["予測ログ"])
    backtest_metrics = BacktestMetrics(
        total_return=0.12,
        win_rate=0.6,
        profit_factor=1.5,
        max_drawdown=0.08,
        gross_profit=0.2,
        gross_loss=0.133,
    )
    trade_log = pd.DataFrame({"decision": [1], "net_return": [0.05]})
    equity_curve = pd.Series([1.0, 1.05])
    backtest_result = BacktestResult(
        metrics=backtest_metrics,
        trade_log=trade_log,
        equity_curve=equity_curve,
        messages=["バックテストログ"],
    )
    summary = ExecutionSummary(
        messages=["処理完了メッセージ"],
        evaluation_result=evaluation_result,
        training_result=training_result,
        prediction_result=prediction_result,
        backtest_result=backtest_result,
        export_path=tmp_path / "trades.csv",
        chart_dataframe=prediction_outputs,
    )

    def fake_execute(
        _: pd.DataFrame,
        progress_callback=None,
        log_callback=None,
    ) -> ExecutionSummary:
        if progress_callback is not None:
            progress_callback(100)
        if log_callback is not None:
            log_callback("処理完了メッセージ")
        return summary

    monkeypatch.setattr(window, "_execute_pipeline", fake_execute)

    def fake_info(parent, title, message):  # type: ignore[override]
        captured["info"] = message

    monkeypatch.setattr("ui.main_window.QMessageBox.information", fake_info)

    file_path = tmp_path / "sample.csv"
    file_path.write_text("dummy")
    window.file_path_input.setText(str(file_path))

    window._on_execute()

    assert captured["info"] == "処理が完了しました。"
    assert window.progress_bar.value() == 100
    assert "処理完了メッセージ" in window.log_output.toPlainText()
    expected_eval = window._format_evaluation_summary(evaluation_result)
    assert window.evaluation_label.text() == expected_eval
    assert f"{training_result.accuracy:.3f}" in window.training_label.text()
    assert str(len(prediction_outputs)) in window.prediction_label.text()
    assert str(summary.export_path) in window.backtest_label.text()


def test_prepare_learning_dataset_aligns_returns(qtbot: QtBot) -> None:
    """学習用データセット整形でターゲットとリターンが揃うことを確認する。"""

    window = _create_window(qtbot)

    base_df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "Open": [100, 101, 100, 102, 101, 103],
            "High": [101, 102, 101, 103, 102, 104],
            "Low": [99, 100, 99, 101, 100, 102],
            "Close": [100, 101, 100.5, 101, 99.5, 100.5],
            "MA5": [100, 100.5, 100.6, 100.8, 100.7, 100.9],
            "MA25": [100, 100.1, 100.2, 100.3, 100.4, 100.5],
            "MA75": [100, 100.05, 100.1, 100.15, 100.2, 100.25],
        }
    )

    features = pd.DataFrame({"feature_a": range(len(base_df))})

    prepared_features, target, actual_returns = window._prepare_learning_dataset(base_df, features)

    assert len(prepared_features) == len(base_df) - 1
    assert prepared_features.index.equals(target.index)
    assert actual_returns.index.equals(target.index)
    assert set(target.unique()) <= {0, 1}
    assert actual_returns.notna().all()
