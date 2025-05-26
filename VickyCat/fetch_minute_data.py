# fetch_minute_data.py
# 支持从 2023-12-04 开始抓取 1 分钟 K 线数据，按 symbol 分表保存

import sys
import os
import sqlite3
from datetime import datetime, timedelta, date
from longport.openapi import QuoteContext, Config, Period, AdjustType
from typing import List
import time

DB_PATH = "minute_data.db"
START_DATE = date(2023, 12, 4)
PERIOD = Period.Min_1
ADJUST_TYPE = AdjustType.NoAdjust
TABLE_PREFIX = "minute_"

config = Config.from_env()
ctx = QuoteContext(config)


def ensure_table(conn: sqlite3.Connection, symbol: str):
    table = TABLE_PREFIX + symbol.replace(".", "_")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            timestamp TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            turnover REAL
        )
    """)
    conn.commit()
    return table


def get_latest_timestamp(conn: sqlite3.Connection, table: str) -> str:
    cursor = conn.execute(f"SELECT MAX(timestamp) FROM {table}")
    row = cursor.fetchone()
    ts = row[0] if row and row[0] is not None else ""
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts



def fetch_and_save_symbol(symbol: str, conn: sqlite3.Connection):
    table = ensure_table(conn, symbol)
    latest_ts_str = get_latest_timestamp(conn, table)

    today = date.today()
    cur = START_DATE
    if latest_ts_str:
        if isinstance(latest_ts_str, datetime):
            dt_latest = latest_ts_str
        else:
            dt_latest = datetime.strptime(latest_ts_str, "%Y-%m-%d %H:%M:%S")
        cur = dt_latest.date() + timedelta(days=1)

    print(f"Fetching {symbol} from {cur} to {today}")

    while cur <= today:
        try:
            resp = ctx.history_candlesticks_by_date(symbol, PERIOD, ADJUST_TYPE, cur, cur)
            candles = resp
            if not candles:
                cur += timedelta(days=1)
                continue

            print(f"{symbol} - {cur} got {len(candles)} records")
            for item in candles:
                timestamp = item.timestamp
                if isinstance(timestamp, str):
                    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                elif isinstance(timestamp, datetime):
                    dt = timestamp
                else:
                    raise ValueError(f"Unexpected timestamp type: {type(timestamp)}")

                formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")  # 格式化成字符串存数据库
                conn.execute(
                    f"INSERT OR IGNORE INTO {table} (timestamp, open, high, low, close, volume, turnover) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        formatted_ts,
                        float(item.open),
                        float(item.high),
                        float(item.low),
                        float(item.close),
                        int(item.volume),
                        float(item.turnover)
                    )
                )
            conn.commit()
        except Exception as e:
            print(f"!! Failed to fetch {cur}: {e}")
        cur += timedelta(days=1)
        time.sleep(0.2)




def main():
    symbols = sys.argv[1:]
    if not symbols:
        symbols = ["TSLA.US", "TSDD.US"]

    with sqlite3.connect(DB_PATH, detect_types=0) as conn:
        for symbol in symbols:
            fetch_and_save_symbol(symbol, conn)


if __name__ == "__main__":
    main()
