"""PySide6で為替予測ツールのUIモックを構築するモジュール。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame
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
        """入力値の検証を実行する。"""
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

        self._show_info("入力データの検証が完了しました。")

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