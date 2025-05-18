import os
import sqlite3
from typing import Dict, Any, List
from datetime import datetime, timedelta

DB_PATH = "market_data_async.db"

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.create_tables()

    def create_tables(self):
        """初始化 quotes 表结构并设置复合主键"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                timestamp TEXT,
                symbol TEXT,
                sequence INTEGER,
                price REAL,
                volume INTEGER,
                turnover REAL,
                PRIMARY KEY (timestamp, symbol, sequence)
            )
        ''')
        self.conn.commit()

    def insert_quote(self, quote_data: Dict[str, Any]):
        """插入逐笔行情数据，按秒生成 sequence"""
        cursor = self.conn.cursor()
        try:
            # 获取当前秒内的最大 sequence 值
            cursor.execute('''
                SELECT MAX(sequence) FROM quotes 
                WHERE timestamp = ? AND symbol = ?
            ''', (quote_data['timestamp'], quote_data['symbol']))
            max_sequence = cursor.fetchone()[0]
            sequence = (max_sequence + 1) if max_sequence is not None else 1

            cursor.execute('''
                INSERT INTO quotes (timestamp, symbol, sequence, price, volume, turnover)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                quote_data['timestamp'],
                quote_data['symbol'],
                sequence,
                quote_data['price'],
                quote_data['volume'],
                quote_data['turnover']
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"插入数据出错: {e}")
            self.conn.rollback()

    def get_kline_1s(self, start_time: str, end_time: str, symbol: str) -> List[Dict[str, Any]]:
        """按需生成 1 秒 K 线数据"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, MIN(price) AS open, MAX(price) AS high, MIN(price) AS low, MAX(price) AS close, 
                   SUM(volume) AS volume, SUM(turnover) AS turnover
            FROM quotes
            WHERE timestamp BETWEEN ? AND ? AND symbol = ?
            GROUP BY timestamp
        ''', (start_time, end_time, symbol))
        data = cursor.fetchall()
        kline_data = [
            {
                "timestamp": row[0], "open": row[1], "high": row[2], "low": row[3],
                "close": row[4], "volume": row[5], "turnover": row[6]
            }
            for row in data
        ]
        return kline_data

    def archive_old_data(self):
        """归档3天前数据并删除"""
        cursor = self.conn.cursor()
        archive_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # 归档数据
        archive_path = f"archive_{archive_date}.db"
        if not os.path.exists(archive_path):
            archive_conn = sqlite3.connect(archive_path)
            archive_conn.execute('''
                CREATE TABLE IF NOT EXISTS quotes (
                    timestamp TEXT,
                    symbol TEXT,
                    sequence INTEGER,
                    price REAL,
                    volume INTEGER,
                    turnover REAL,
                    PRIMARY KEY (timestamp, symbol, sequence)
                )
            ''')
            archive_conn.commit()
            archive_conn.close()

        cursor.execute('''
            INSERT INTO quotes SELECT * FROM main.quotes WHERE timestamp < ?
        ''', (archive_date,))

        # 删除3天前数据
        cursor.execute('''
            DELETE FROM quotes WHERE timestamp < ?
        ''', (archive_date,))

        self.conn.commit()

async def save_quotes_batch(self, quote_data_list: List[Dict[str, Any]]):
    """批量保存逐笔行情数据"""
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 批量插入
        cursor.executemany('''
            INSERT INTO quotes (timestamp, symbol, sequence, price, volume, turnover)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', [
            (
                data["timestamp"],
                data["symbol"],
                self.get_next_sequence(data["timestamp"], data["symbol"]),
                data["price"],
                data["volume"],
                data["turnover"]
            ) for data in quote_data_list
        ])

        conn.commit()

    except Exception as e:
        print(f"[Error] 批量插入数据失败: {e}")
        conn.rollback()

    finally:
        if conn:
            conn.close()