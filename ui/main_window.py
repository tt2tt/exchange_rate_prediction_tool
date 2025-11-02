"""PySide6で為替予測ツールのUIモックを構築するモジュール。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from pandas import DataFrame, Series
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.data.validator import DataValidator
from core.data.preprocessing.pipeline import PreprocessingPipeline
from core.features import FeatureEngineer
from core.model import (
    BacktestCSVExporter,
    BacktestConfig,
    Backtester,
    EvaluationConfig,
    EvaluationResult,
    LogisticRegressionTrainer,
    ModelEvaluator,
    Predictor,
)


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
        try:
            # 全処理フローを実行し、結果メッセージを取得する
            summary = self._execute_pipeline(dataframe)
        except ValueError as exc:
            # 想定済みの入力不備はValueErrorとして扱い利用者へ通知する
            self._show_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            # 予期しない例外は詳細を含めたエラーとして通知する
            self._show_error(f"処理中に予期せぬエラーが発生しました: {exc}")
            return

        # 正常終了時は統合メッセージを情報ダイアログで通知する
        self._show_info(summary)

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
        return pd.read_csv(path)

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

    def _execute_pipeline(self, dataframe: DataFrame) -> str:
        """前処理からバックテストまでの一連処理を実行する。"""
        # 処理経過を蓄積し、最終的にユーザーへまとめて提示する
        messages: list[str] = ["入力データの検証が完了しました。"]

        # 前処理パイプラインでソートや検証整理を行う
        preprocessing = PreprocessingPipeline(self._validator)
        preprocessing_result = preprocessing.preprocess(dataframe)
        cleaned_df = preprocessing_result.dataframe
        messages.extend(preprocessing_result.messages)

        # 特徴量生成モジュールで学習用特徴量を構築する
        feature_engineer = FeatureEngineer(self._validator)
        lag = 1
        feature_result = feature_engineer.generate(cleaned_df, lag=lag)
        features = feature_result.features
        messages.extend(feature_result.messages)
        if features.empty:
            raise ValueError("特徴量生成に失敗しました。データのカラム構成を確認してください。")

        # 学習に利用する目的変数と実績リターンを整形する
        learning_features, target, actual_returns = self._prepare_learning_dataset(cleaned_df, features)
        if target.empty:
            raise ValueError("学習用データが空です。入力データ件数を確認してください。")
        if target.nunique(dropna=False) < 2:
            raise ValueError("目的変数が単一クラスのため学習できません。データ内容を見直してください。")

        # 時系列分割でモデルの汎化性能を確認する
        evaluation_config = self._build_evaluation_config(len(learning_features))
        evaluator = ModelEvaluator(LogisticRegressionTrainer(), evaluation_config)
        evaluation_result = evaluator.evaluate(learning_features, target)
        messages.extend(evaluation_result.messages)
        messages.append(self._format_evaluation_summary(evaluation_result))

        # 全データで最終モデルを学習する
        trainer = LogisticRegressionTrainer()
        training_result = trainer.train(learning_features, target)
        messages.extend(training_result.messages)

        # 学習済みモデルで推論結果を生成する
        predictor = Predictor(training_result=training_result)
        prediction_result = predictor.predict(learning_features)
        messages.extend(prediction_result.messages)

        # 予測シグナルを用いてバックテストを実行する
        backtest_input = pd.DataFrame(
            {
                "decision": prediction_result.outputs["decision"],
                "actual_return": actual_returns.loc[prediction_result.outputs.index],
            }
        )
        backtester = Backtester(BacktestConfig(spread=float(self.spread_input.value())))
        backtest_result = backtester.run(backtest_input)
        messages.extend(backtest_result.messages)

        # バックテスト結果をCSVに書き出し保存先を報告する
        exporter = BacktestCSVExporter()
        export_path = exporter.export(backtest_result)
        messages.append(f"バックテスト結果を{export_path}へ出力しました。")

        return "\n".join(messages)

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
