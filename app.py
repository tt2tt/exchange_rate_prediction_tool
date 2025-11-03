"""PySide6アプリを起動するエントリポイント。"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> None:
    """為替予測ツールのGUIアプリケーションを起動する。

    Raises:
        SystemExit: Qtのイベントループが終了コードを返すときに送出される。
    """
    # QApplicationを初期化し、Qtアプリ全体のライフサイクルを管理する
    app = QApplication(sys.argv)
    # メインウィンドウを生成し、ユーザーにUIを提示する
    window = MainWindow()
    window.show()
    # Qtのイベントループへ制御を渡してアプリを継続させる
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
