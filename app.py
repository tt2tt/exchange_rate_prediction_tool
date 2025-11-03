"""PySide6アプリを起動するエントリポイント。"""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow

def _apply_dark_theme(app: QApplication) -> None:
    """アプリ全体へダークテーマのパレットとスタイルを適用する。

    Args:
        app: Qtアプリケーション本体。
    """
    # Fusionスタイルを指定し、配色変更を行いやすくする
    app.setStyle("Fusion")

    # メインウィンドウや背景の基調色を暗色へ揃える
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.Base, QColor(24, 24, 24))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(36, 36, 36))
    palette.setColor(QPalette.ColorRole.Text, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # ラベルやグループボックスの枠線色を整えて視認性を確保する
    app.setStyleSheet(
        """
        QWidget { background-color: #1E1E1E; color: #E5E5E5; }
        QGroupBox { border: 1px solid #3C3C3C; margin-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        QPushButton { background-color: #2D2D2D; border: 1px solid #4A4A4A; padding: 6px; }
        QPushButton:hover { background-color: #3A3A3A; }
        QLineEdit, QTextEdit { background-color: #252525; border: 1px solid #3F3F3F; }
        QProgressBar { background-color: #252525; border: 1px solid #3F3F3F; text-align: center; }
        QProgressBar::chunk { background-color: #0078D4; }
        QDoubleSpinBox { background-color: #252525; border: 1px solid #3F3F3F; }
        """
    )


def main() -> None:
    """為替予測ツールのGUIアプリケーションを起動する。

    Raises:
        SystemExit: Qtのイベントループが終了コードを返すときに送出される。
    """
    # QApplicationを初期化し、Qtアプリ全体のライフサイクルを管理する
    app = QApplication(sys.argv)
    # ダークテーマを適用してデザインポリシーに合わせる
    _apply_dark_theme(app)
    # メインウィンドウを生成し、ユーザーにUIを提示する
    window = MainWindow()
    window.show()
    # Qtのイベントループへ制御を渡してアプリを継続させる
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
