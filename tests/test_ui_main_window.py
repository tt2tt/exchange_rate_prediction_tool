"""UIメインウィンドウのモックを検証するテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytestqt.qtbot import QtBot  # type: ignore[import-not-found]
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QPushButton
from sklearn.pipeline import Pipeline

from core.model.backtest_csv_exporter import BacktestCSVExporter
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


def _build_sample_dataframe(row_count: int = 80) -> pd.DataFrame:
    """パイプライン検証用のダミー為替データを生成する。"""
    # テスト用に上昇と下落が交互に現れる価格系列を作成する
    dates = pd.date_range("2024-01-01", periods=row_count, freq="D")
    pattern = np.array([1.5, -1.2, 1.0, -1.0], dtype=float)
    increments = np.resize(pattern, row_count)
    close = 100.0 + np.cumsum(increments)
    open_ = close - (increments * 0.2)
    high = close + 0.6
    low = close - 0.8

    # 移動平均は欠損を出さないよう最小期間を1に設定する
    close_series = pd.Series(close)
    ma5 = close_series.rolling(window=5, min_periods=1).mean()
    ma25 = close_series.rolling(window=25, min_periods=1).mean()
    ma75 = close_series.rolling(window=45, min_periods=1).mean()

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "MA5": ma5,
            "MA25": ma25,
            "MA75": ma75,
        }
    )


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


def test_execute_pipeline_generates_chart_dataframe(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """パイプライン実行でチャート描画用データが生成されることを確認する。"""

    window = _create_window(qtbot)

    # バックテストCSV出力は一時パスへ書き出すよう差し替える
    def _fake_export(self: BacktestCSVExporter, result, output_path=None):  # type: ignore[override]
        export_file = tmp_path / "trades.csv"
        export_file.write_text("entry_index,exit_index,direction,net_return,duration\n", encoding="utf-8")
        return export_file

    monkeypatch.setattr(BacktestCSVExporter, "export", _fake_export, raising=False)

    dataframe = _build_sample_dataframe()

    summary = window._execute_pipeline(dataframe)

    # チャート用DataFrameが空でなく、主要カラムを含むことを検証する
    assert not summary.chart_dataframe.empty
    assert {"Close", "probability_up", "decision"}.issubset(summary.chart_dataframe.columns)
    assert summary.chart_dataframe.index.equals(summary.prediction_result.outputs.index)

    # 結果表示を更新した際にチャート描画が行われることを確認する
    window._display_results(summary)
    assert len(window.chart_figure.axes) == 1
    axis = window.chart_figure.axes[0]
    assert any(line.get_label() == "終値" for line in axis.get_lines())
    assert any(collection.get_label() == "シグナル" for collection in axis.collections)


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
