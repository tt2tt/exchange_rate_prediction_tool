"""データバリデーションに用いる定数。"""

# アプリ内部で想定するカラム名
REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "MA5",
    "MA25",
    "MA75",
]

# CSVに含まれる可能性がある別名 -> 内部名のマッピング
COLUMN_ALIASES = {
    "日付": "Date",
    "始値": "Open",
    "高値": "High",
    "安値": "Low",
    "終値": "Close",
    "期間A[5](日足)": "MA5",
    "期間C[25](日足)": "MA25",
    "期間G[75](日足)": "MA75",
}
