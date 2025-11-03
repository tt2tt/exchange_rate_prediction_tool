"""PySide6で為替予測ツールのUIモックを構築するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from pandas import DataFrame, Series
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.data.validator import DataValidator
from core.data.preprocessing.pipeline import PreprocessingPipeline
from core.features import FeatureEngineer
from core.model import (
    BacktestCSVExporter,
    BacktestConfig,
    Backtester,
    BacktestResult,
    EvaluationConfig,
    EvaluationResult,
    LogisticRegressionTrainer,
    ModelEvaluator,
    Predictor,
    TrainingResult,
    PredictionResult,
)


@dataclass(frozen=True)
class ExecutionSummary:
    """パイプライン処理の要約情報をまとめるデータクラス。"""

    messages: list[str]
    evaluation_result: EvaluationResult
    training_result: TrainingResult
    prediction_result: PredictionResult
    backtest_result: BacktestResult
    export_path: Path
    chart_dataframe: DataFrame


class MainWindow(QMainWindow):
    """アプリケーション全体をまとめるメインウィンドウ。"""

    def __init__(self) -> None:
        """ウィンドウとUIコンポーネントを初期化する。"""
        super().__init__()
        # ウィンドウタイトルで用途を明示する
        self.setWindowTitle("為替予測ツール")
        # 主要ウィジェットを生成する
        self._central_widget: QWidget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._root_layout = QVBoxLayout()
        self._central_widget.setLayout(self._root_layout)

        # データ検証とメッセージ管理用の内部状態を初期化する
        self._validator = DataValidator()
        self._min_rows = 10
        self._last_error_message: Optional[str] = None
        self._last_info_message: Optional[str] = None

        # 各種入力エリアを構築する
        self._build_file_input_section()
        self._build_parameter_section()
        self._build_execute_section()
        self._build_feedback_section()

        # UIイベントをセットアップする
        self._connect_signals()

    def _build_file_input_section(self) -> None:
        """ファイル選択エリアを組み立てる。"""
        # ファイル指定用のグループボックスを設置する
        group_box = QGroupBox("入力データ")
        group_box.setObjectName("fileGroup")
        layout = QHBoxLayout()
        group_box.setLayout(layout)

        # 選択中パスを表示するラインエディット
        label = QLabel("学習データ")
        layout.addWidget(label)

        self.file_path_input = QLineEdit()
        self.file_path_input.setObjectName("filePathInput")
        self.file_path_input.setReadOnly(True)
        layout.addWidget(self.file_path_input, stretch=1)

        # ファイル選択ボタンを配置する
        self.file_select_button = QPushButton("...")
        self.file_select_button.setObjectName("fileSelectButton")
        layout.addWidget(self.file_select_button)

        self._root_layout.addWidget(group_box)

    def _build_parameter_section(self) -> None:
        """数値入力などのパラメータエリアを構築する。"""
        group_box = QGroupBox("バックテスト設定")
        group_box.setObjectName("parameterGroup")
        layout = QHBoxLayout()
        group_box.setLayout(layout)

        # スプレッド設定用の入力欄を用意する
        label = QLabel("スプレッド")
        layout.addWidget(label)

        self.spread_input = QDoubleSpinBox()
        self.spread_input.setObjectName("spreadInput")
        self.spread_input.setDecimals(4)
        self.spread_input.setSingleStep(0.0001)
        self.spread_input.setRange(0.0001, 0.01)
        self.spread_input.setValue(0.0005)
        layout.addWidget(self.spread_input)

        self._root_layout.addWidget(group_box)

    def _build_execute_section(self) -> None:
        """実行ボタンなどの操作エリアを構築する。"""
        # 実行セクションは右寄せボタンを想定する
        container = QWidget()
        layout = QHBoxLayout()
        container.setLayout(layout)

        layout.addStretch(1)

        self.execute_button = QPushButton("実行")
        self.execute_button.setObjectName("executeButton")
        self.execute_button.setEnabled(True)
        layout.addWidget(self.execute_button, alignment=Qt.AlignmentFlag.AlignRight)

        self._root_layout.addWidget(container)

    def _build_feedback_section(self) -> None:
        """進捗バーやログ、結果表示を構築する。"""
        # 進捗バー表示用のグループボックスを用意する
        progress_group = QGroupBox("進捗状況")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # ログ出力領域を構築する
        log_group = QGroupBox("処理ログ")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)

        # サマリー結果を表示するラベルをまとめる
        result_group = QGroupBox("結果サマリー")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)

        self.evaluation_label = QLabel("評価結果は未実行です。")
        self.evaluation_label.setWordWrap(True)
        result_layout.addWidget(self.evaluation_label)

        self.training_label = QLabel("学習結果は未実行です。")
        self.training_label.setWordWrap(True)
        result_layout.addWidget(self.training_label)

        self.prediction_label = QLabel("予測結果は未実行です。")
        self.prediction_label.setWordWrap(True)
        result_layout.addWidget(self.prediction_label)

        self.backtest_label = QLabel("バックテスト結果は未実行です。")
        self.backtest_label.setWordWrap(True)
        result_layout.addWidget(self.backtest_label)

        # チャート表示用の領域を構築する
        chart_group = QGroupBox("ミニチャート")
        chart_layout = QVBoxLayout()
        chart_group.setLayout(chart_layout)

        self.chart_figure = Figure(figsize=(5.0, 3.0), tight_layout=True)
        self.chart_canvas = FigureCanvasQTAgg(self.chart_figure)
        chart_layout.addWidget(self.chart_canvas)

        self._root_layout.addWidget(progress_group)
        self._root_layout.addWidget(log_group)
        self._root_layout.addWidget(result_group)
        self._root_layout.addWidget(chart_group)
        self._root_layout.addStretch(1)

    def _connect_signals(self) -> None:
        """UIシグナルとスロットを連携する。"""
        # ファイル選択と実行ボタンにハンドラを紐付ける
        self.file_select_button.clicked.connect(self._on_select_file)
        self.execute_button.clicked.connect(self._on_execute)

    def _on_select_file(self) -> None:
        """CSVファイル選択ダイアログを表示する。"""
        # ファイル選択ダイアログを開き、選択されたパスを保持する
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "学習データを選択",
            "",
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if file_path:
            self.file_path_input.setText(file_path)

    def _on_execute(self) -> None:
        """入力値の検証と実行フローを起動する。"""
        # ファイルパス未設定の場合は即座にエラーを通知する
        file_path = self.file_path_input.text().strip()
        if not file_path:
            self._show_error("学習データが選択されていません。ファイルを指定してください。")
            return

        path = Path(file_path)
        if not path.is_file():
            self._show_error("指定されたファイルが存在しません。パスを確認してください。")
            return

        try:
            dataframe = self._read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:  # type: ignore[attr-defined]
            self._show_error(f"CSVの読み込みに失敗しました: {exc}")
            return

        errors = self._validate_dataframe(dataframe)
        if errors:
            self._show_error("\n".join(errors))
            return

        # フィードバックエリアを初期化して進捗表示をリセットする
        self._reset_feedback()

        try:
            summary = self._execute_pipeline(
                dataframe,
                progress_callback=self._set_progress_value,
                log_callback=self._append_log,
            )
        except ValueError as exc:
            # 想定済みの入力不備はValueErrorとして扱い利用者へ通知する
            self._show_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            # 予期しない例外は詳細を含めたエラーとして通知する
            self._show_error(f"処理中に予期せぬエラーが発生しました: {exc}")
            return

        # 結果表示を更新し、完了メッセージを通知する
        self._display_results(summary)
        self._show_info("処理が完了しました。")

    def _validate_dataframe(self, dataframe: DataFrame) -> list[str]:
        """DataFrameを検証し、問題があればメッセージを返す。"""
        messages: list[str] = []

        validation = self._validator.run_all(dataframe)
        messages.extend(validation.messages)

        # 行数が所定の閾値に満たない場合はエラーとみなす
        if len(dataframe) < self._min_rows:
            messages.append(f"データ件数が不足しています。最小{self._min_rows}行以上が必要です。")

        # スプレッド値が正の範囲に収まっているかを最終確認する
        if self.spread_input.value() <= 0.0:
            messages.append("スプレッドは0より大きい値を設定してください。")

        return messages

    def _read_csv(self, path: Path) -> DataFrame:
        """CSVファイルをDataFrameとして読み込む。"""
        # 主要サポートエンコーディングを優先順位で試行する
        encodings = ("utf-8", "cp932")
        last_error: Optional[Exception] = None

        for encoding in encodings:
            try:
                # エンコーディングを指定して読み込みを試みる
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as exc:
                # 読み込み失敗時は次の候補で再試行する
                last_error = exc
                continue

        # いずれのエンコーディングでも失敗した場合はValueErrorで通知する
        raise ValueError(f"サポート外のエンコーディングです: {last_error}")

    def _show_error(self, message: str) -> None:
        """エラーメッセージをダイアログで通知する。"""
        self._last_error_message = message
        self._last_info_message = None
        QMessageBox.critical(self, "入力エラー", message)

    def _show_info(self, message: str) -> None:
        """情報メッセージをダイアログで通知する。"""
        self._last_info_message = message
        self._last_error_message = None
        QMessageBox.information(self, "完了", message)

    def _reset_feedback(self) -> None:
        """進捗バーやログ、結果表示を初期状態へ戻す。"""
        self._set_progress_value(0)
        self.log_output.clear()
        self.evaluation_label.setText("評価結果は未実行です。")
        self.training_label.setText("学習結果は未実行です。")
        self.prediction_label.setText("予測結果は未実行です。")
        self.backtest_label.setText("バックテスト結果は未実行です。")
        self.chart_figure.clear()
        self.chart_canvas.draw()

    def _set_progress_value(self, value: int) -> None:
        """プログレスバーを更新し、UI反映を促す。"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def _append_log(self, message: str) -> None:
        """ログ表示領域へメッセージを追記する。"""
        self.log_output.append(message)

    def _display_results(self, summary: ExecutionSummary) -> None:
        """処理結果サマリーを結果ラベルへ反映する。"""
        evaluation_text = self._format_evaluation_summary(summary.evaluation_result)
        self.evaluation_label.setText(evaluation_text)

        training_text = (
            f"学習精度: {summary.training_result.accuracy:.3f}"
            f" / モデル保存先: {summary.training_result.model_path}"
        )
        self.training_label.setText(training_text)

        prediction_text = self._format_prediction_summary(summary.prediction_result)
        self.prediction_label.setText(prediction_text)

        backtest_text = self._format_backtest_summary(summary.backtest_result, summary.export_path)
        self.backtest_label.setText(backtest_text)
        # チャート描画用DataFrameが渡されている場合はミニチャートを更新する
        self._render_chart(summary.chart_dataframe)

    def _execute_pipeline(
        self,
        dataframe: DataFrame,
        progress_callback: Optional[Callable[[int], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> ExecutionSummary:
        """前処理からバックテストまでの一連処理を実行する。"""
        messages: list[str] = []

        def _record(message: str, progress: Optional[int] = None) -> None:
            """内部ログを蓄積しつつUIへフィードバックする。"""
            messages.append(message)
            if log_callback is not None:
                log_callback(message)
            if progress_callback is not None and progress is not None:
                progress_callback(progress)

        _record("入力データの検証が完了しました。", progress=5)

        preprocessing = PreprocessingPipeline(self._validator)
        _record("前処理を実行します。", progress=10)
        preprocessing_result = preprocessing.preprocess(dataframe)
        cleaned_df = preprocessing_result.dataframe
        for message in preprocessing_result.messages:
            _record(message)
        _record("前処理が完了しました。", progress=20)

        feature_engineer = FeatureEngineer(self._validator)
        _record("特徴量を生成します。", progress=25)
        feature_result = feature_engineer.generate(cleaned_df, lag=1)
        features = feature_result.features
        for message in feature_result.messages:
            _record(message)
        if features.empty:
            raise ValueError("特徴量生成に失敗しました。データのカラム構成を確認してください。")
        _record("特徴量の生成が完了しました。", progress=35)

        learning_features, target, actual_returns = self._prepare_learning_dataset(cleaned_df, features)
        if target.empty:
            raise ValueError("学習用データが空です。入力データ件数を確認してください。")
        if target.nunique(dropna=False) < 2:
            raise ValueError("目的変数が単一クラスのため学習できません。データ内容を見直してください。")
        _record("学習用データセットを整形しました。", progress=45)

        evaluation_config = self._build_evaluation_config(len(learning_features))
        _record("時系列交差検証を実行します。", progress=55)
        evaluator = ModelEvaluator(LogisticRegressionTrainer(), evaluation_config)
        evaluation_result = evaluator.evaluate(learning_features, target)
        for message in evaluation_result.messages:
            _record(message)
        evaluation_summary = self._format_evaluation_summary(evaluation_result)
        _record(evaluation_summary, progress=65)

        trainer = LogisticRegressionTrainer()
        _record("全データで最終モデルを学習します。", progress=70)
        training_result = trainer.train(learning_features, target)
        for message in training_result.messages:
            _record(message)
        _record("モデル学習が完了しました。", progress=80)

        predictor = Predictor(training_result=training_result)
        _record("学習済みモデルで推論します。", progress=85)
        prediction_result = predictor.predict(learning_features)
        for message in prediction_result.messages:
            _record(message)

        # チャート描画用に予測結果と終値をまとめたDataFrameを構築する
        chart_dataframe = cleaned_df.loc[prediction_result.outputs.index].copy()
        chart_dataframe["probability_up"] = prediction_result.outputs["probability_up"]
        chart_dataframe["decision"] = prediction_result.outputs["decision"]
        if "decision_label" in prediction_result.outputs:
            chart_dataframe["decision_label"] = prediction_result.outputs["decision_label"]

        backtest_input = pd.DataFrame(
            {
                "decision": prediction_result.outputs["decision"],
                "actual_return": actual_returns.loc[prediction_result.outputs.index],
            }
        )
        backtester = Backtester(BacktestConfig(spread=float(self.spread_input.value())))
        _record("バックテストを実行します。", progress=90)
        backtest_result = backtester.run(backtest_input)
        for message in backtest_result.messages:
            _record(message)

        exporter = BacktestCSVExporter()
        export_path = exporter.export(backtest_result)
        _record(f"バックテスト結果を{export_path}へ出力しました。", progress=100)

        return ExecutionSummary(
            messages=messages,
            evaluation_result=evaluation_result,
            training_result=training_result,
            prediction_result=prediction_result,
            backtest_result=backtest_result,
            export_path=export_path,
            chart_dataframe=chart_dataframe,
        )

    def _prepare_learning_dataset(self, dataframe: DataFrame, features: DataFrame) -> Tuple[DataFrame, Series, Series]:
        """特徴量とターゲット・実績リターンをアライメントする。"""
        # Close列から翌日のリターンを算出し、上昇可否をターゲットに変換する
        close = dataframe["Close"].astype(float)
        next_close = close.shift(-1)
        forward_return = (next_close - close) / close
        target = (forward_return > 0).astype(int)

        # 特徴量・ターゲット・リターンを結合して欠損行を除去する
        merged = features.copy()
        merged["target"] = target
        merged["actual_return"] = forward_return
        merged = merged.dropna()

        # 欠損除去後のデータを学習用に分割して返す
        prepared_features = merged.drop(columns=["target", "actual_return"])
        prepared_target = merged["target"].astype(int)
        prepared_returns = merged["actual_return"].astype(float)
        return prepared_features, prepared_target, prepared_returns

    def _build_evaluation_config(self, sample_size: int) -> EvaluationConfig:
        """データ件数に応じてTimeSeriesSplit設定を調整する。"""
        # 分割数はサンプル件数に応じて上限を抑え、最低2分割を確保する
        splits = max(2, min(5, max(2, sample_size // 4)))
        return EvaluationConfig(n_splits=min(splits, max(2, sample_size - 1)))

    def _format_evaluation_summary(self, result: EvaluationResult) -> str:
        """評価指標からユーザー向けサマリー文字列を生成する。"""
        # NaNを含む可能性があるため安全に文字列化する補助関数を定義する
        def _fmt(value: float) -> str:
            return "nan" if pd.isna(value) else f"{value:.3f}"

        metrics = result.overall_metrics
        return (
            "評価結果: accuracy={acc}, precision={prec}, recall={rec}, roc_auc={auc}, expected={exp}".format(
                acc=_fmt(metrics.get("accuracy", float("nan"))),
                prec=_fmt(metrics.get("precision", float("nan"))),
                rec=_fmt(metrics.get("recall", float("nan"))),
                auc=_fmt(metrics.get("roc_auc", float("nan"))),
                exp=_fmt(metrics.get("expected_value", float("nan"))),
            )
        )

    def _format_prediction_summary(self, result: PredictionResult) -> str:
        """予測結果の統計をテキスト化する。"""
        outputs = result.outputs
        total = len(outputs)
        if total == 0:
            return "予測結果は空です。"

        positive_rate = float(outputs["decision"].mean()) if "decision" in outputs else 0.0
        average_prob = float(outputs["probability_up"].mean()) if "probability_up" in outputs else 0.0

        return (
            "予測件数: {count} / 上昇判定率: {rate:.3f} / 平均上昇確率: {prob:.3f}".format(
                count=total,
                rate=positive_rate,
                prob=average_prob,
            )
        )

    def _format_backtest_summary(self, result: BacktestResult, export_path: Path) -> str:
        """バックテスト指標をテキスト化する。"""
        metrics = result.metrics
        return (
            "総リターン: {ret:.4f} / 勝率: {win:.3f} / PF: {pf:.3f} / 最大DD: {dd:.4f} / CSV: {path}".format(
                ret=metrics.total_return,
                win=metrics.win_rate,
                pf=metrics.profit_factor if pd.notna(metrics.profit_factor) else float("nan"),
                dd=metrics.max_drawdown,
                path=export_path,
            )
        )

    def _render_chart(self, chart_dataframe: DataFrame) -> None:
        """ミニチャートへ終値・移動平均・シグナルを描画する。"""
        # 描画前に既存の内容をクリアして毎回クリーンな状態で描く
        self.chart_figure.clear()
        axis = self.chart_figure.add_subplot(1, 1, 1)
        axis.set_facecolor("#202020")

        # データが空の場合はメッセージのみ表示して処理を終了する
        if chart_dataframe.empty:
            axis.text(0.5, 0.5, "チャート表示用のデータがありません。", color="#E5E5E5", ha="center", va="center")
            self.chart_canvas.draw()
            return

        # 日付列が存在すれば時系列軸として利用し、無ければインデックスで代替する
        x_axis = (
            pd.to_datetime(chart_dataframe["Date"], errors="coerce")
            if "Date" in chart_dataframe.columns
            else chart_dataframe.index
        )

        # 終値ラインを描画して基準となる価格推移を表示する
        if "Close" in chart_dataframe.columns:
            axis.plot(x_axis, chart_dataframe["Close"].astype(float), label="終値", color="#4FC3F7")

        # 移動平均線が存在する場合は補助線として重ねて傾向を把握しやすくする
        for ma_col, color in (("MA5", "#AED581"), ("MA25", "#81C784"), ("MA75", "#4CAF50")):
            if ma_col in chart_dataframe.columns:
                axis.plot(x_axis, chart_dataframe[ma_col].astype(float), label=ma_col, color=color, linewidth=1.0)

        # 売買シグナルが1となる位置にマーカーを描画してエントリーポイントを可視化する
        if "decision" in chart_dataframe.columns and "Close" in chart_dataframe.columns:
            decision_series = chart_dataframe["decision"].fillna(0).astype(int)
            signal_mask = decision_series == 1
            if signal_mask.any():
                axis.scatter(
                    x_axis[signal_mask],
                    chart_dataframe.loc[signal_mask, "Close"].astype(float),
                    marker="^",
                    color="#FFB74D",
                    edgecolor="#FFC107",
                    s=50,
                    label="シグナル",
                )

        # 凡例やグリッドを設定してチャートの読みやすさを向上させる
        axis.set_title("終値とシグナルの推移", color="#E0E0E0")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="upper left", facecolor="#2A2A2A", edgecolor="#444444")
        axis.grid(color="#404040", linestyle="--", linewidth=0.5)
        for label in axis.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        axis.tick_params(colors="#CCCCCC")
        axis.spines["bottom"].set_color("#5A5A5A")
        axis.spines["left"].set_color("#5A5A5A")

        self.chart_canvas.draw()
