"""UIメインウィンドウのモックを検証するテスト。"""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot  # type: ignore[import-not-found]
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QPushButton

from ui.main_window import MainWindow


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


def test_spread_input_has_default_value(qtbot: QtBot) -> None:
    """スプレッド入力欄の初期値と設定を検証する。"""
    window = _create_window(qtbot)
    spread_input = window.findChild(QDoubleSpinBox, "spreadInput")
    assert spread_input is not None

    # 初期値と刻み幅が想定どおりであることを確認する
    assert spread_input.value() == pytest.approx(0.0005)
    assert spread_input.singleStep() == pytest.approx(0.0001)
    assert spread_input.maximum() == pytest.approx(0.01)