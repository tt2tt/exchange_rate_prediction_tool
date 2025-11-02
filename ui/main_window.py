"""PySide6で為替予測ツールのUIモックを構築するモジュール。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
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

        # 各種入力エリアを構築する
        self._build_file_input_section()
        self._build_parameter_section()
        self._build_execute_section()

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
        self.spread_input.setRange(0.0, 0.01)
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