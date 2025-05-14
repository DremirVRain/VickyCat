import asyncio
import sqlite3
from datetime import datetime
from typing import Dict, Any

DB_PATH = "market_data_async.db"

def init_db():
    """初始化 kline_1s 数据表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kline_1s (
            timestamp TEXT,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            turnover REAL,
            volume_diff INTEGER,
            turnover_diff REAL,
            PRIMARY KEY (timestamp, symbol)
        )
    ''')
    conn.commit()
    conn.close()

async def save_kline_to_db(kline_data: Dict[str, Any]):
    """保存1秒K线数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO kline_1s (timestamp, symbol, open, high, low, close, volume, turnover, volume_diff, turnover_diff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kline_data['timestamp'],
            kline_data['symbol'],
            kline_data['open'],
            kline_data['high'],
            kline_data['low'],
            kline_data['close'],
            kline_data['volume'],
            kline_data['turnover'],
            kline_data['volume_diff'],
            kline_data['turnover_diff']
        ))
        conn.commit()
    except Exception as e:
        print(f"数据库插入错误: {e}")
    finally:
        conn.close()

async def start_data_processing():
    """从 quotes 表中提取数据并生成 1 秒 K 线数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    while True:
        try:
            cursor.execute('''
                SELECT timestamp, symbol, price, volume, turnover FROM quotes 
                WHERE timestamp >= (SELECT datetime('now', '-1 seconds'))
            ''')
            rows = cursor.fetchall()

            if not rows:
                await asyncio.sleep(1)
                continue

            symbol_data = {}
            for row in rows:
                symbol = row[1]
                if symbol not in symbol_data:
                    symbol_data[symbol] = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol,
                        "open": row[2],
                        "high": row[2],
                        "low": row[2],
                        "close": row[2],
                        "volume": 0,
                        "turnover": 0,
                        "volume_diff": 0,
                        "turnover_diff": 0,
                    }

                symbol_data[symbol]["high"] = max(symbol_data[symbol]["high"], row[2])
                symbol_data[symbol]["low"] = min(symbol_data[symbol]["low"], row[2])
                symbol_data[symbol]["close"] = row[2]
                symbol_data[symbol]["volume"] += row[3]
                symbol_data[symbol]["turnover"] += row[4]

            for symbol, kline_data in symbol_data.items():
                cursor.execute('''
                    SELECT volume, turnover FROM kline_1s WHERE symbol = ? 
                    ORDER BY timestamp DESC LIMIT 1
                ''', (symbol,))
                last_data = cursor.fetchone()
                prev_volume = last_data[0] if last_data else 0
                prev_turnover = last_data[1] if last_data else 0

                kline_data["volume_diff"] = kline_data["volume"] - prev_volume
                kline_data["turnover_diff"] = kline_data["turnover"] - prev_turnover

                await save_kline_to_db(kline_data)

        except Exception as e:
            print(f"数据处理错误: {e}")
        await asyncio.sleep(1)