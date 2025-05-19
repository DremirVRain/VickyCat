import os
import sqlite3
from typing import Dict, Any, List
from datetime import datetime, timedelta
from config import TRADE_SESSION
from utils.archive_manager import ArchiveManager
from utils.data_cache import DataCache

DB_PATH = "market_data_async.db"

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.create_tables()

        # 初始化 ArchiveManager
        self.archive_manager = ArchiveManager()

        # 初始化缓存
        self.data_cache = DataCache()

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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quote_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                price REAL,
                volume INTEGER,
                turnover REAL,
                error_message TEXT
            )
        ''')
        self.conn.commit()

    def is_trading_session(exchange: str) -> bool:
        """
        检查当前是否为指定交易所的交易时间段内
        :param exchange: 交易所名称 ('US', 'HK', 'CN')
        :return: True 表示当前为交易时间段内，False 表示非交易时间段
        """
        now = datetime.now()
        current_time = now.hour * 100 + now.minute

        # 获取当前交易所的交易时间段
        session = TRADE_SESSION.get(exchange)
        if not session:
            print(f"[Warning] 未定义的交易所：{exchange}")
            return False

        # 遍历交易时段，检查是否处于交易时间段内
        for period, (start, end) in session.items():
            if start <= current_time <= end:
                return True

        return False

    def get_next_sequence(self, timestamp: str, symbol: str) -> int:
        """获取当前秒内的下一个 sequence 值"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(sequence) FROM quotes WHERE timestamp = ? AND symbol = ?
        ''', (timestamp, symbol))
        max_sequence = cursor.fetchone()[0]
        return (max_sequence + 1) if max_sequence is not None else 1

    async def save_quotes_batch(self, quote_data_list: List[Dict[str, Any]]):
        """批量保存逐笔行情数据，支持数据压缩与异常日志记录"""
        if not quote_data_list:
            print("[Info] No data to insert.")
            return

        try:
            cursor = self.conn.cursor()

            # 构建插入数据，新增异常数据缓存
            data_to_insert = []
            error_data = []

            for data in quote_data_list:
                try:
                    if not self.is_trading_session():
                        print(f"[Info] 非交易时段数据忽略: {data}")
                        continue
                    
                    symbol = data["symbol"]
                    timestamp = data["timestamp"]

                    data_to_insert.append((
                        data['timestamp'],
                        data['symbol'],
                        self.get_next_sequence(data['timestamp'], data['symbol']),
                        data['price'],
                        data['volume'],
                        data['turnover']
                    ))

                except KeyError as e:
                    print(f"[Warning] 数据字段缺失: {data} - {e}")
                    error_data.append(data)

            # 插入有效数据
            if data_to_insert:
                cursor.executemany('''
                    INSERT INTO quotes (timestamp, symbol, sequence, price, volume, turnover)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', data_to_insert)
                self.conn.commit()
                print(f"[Info] 成功插入 {len(data_to_insert)} 条数据。")

            # 处理异常数据
            for data, error_msg in error_data:
                self.log_error_data(data, error_msg)

        except Exception as e:
            print(f"[Error] 批量插入数据失败: {e}")
            self.conn.rollback()

    def log_error_data(self, data: Dict[str, Any], error_message: str):
        """记录异常数据到 quote_errors 表"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO quote_errors (timestamp, symbol, price, volume, turnover, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data.get('timestamp', "N/A"),
                data.get('symbol', "N/A"),
                data.get('price', 0.0),
                data.get('volume', 0),
                data.get('turnover', 0.0),
                error_message
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            print(f"[Error] 记录异常数据失败: {e}")
            self.conn.rollback()

    def archive_old_data(self):
        """归档并删除旧数据，通过 ArchiveManager 统一管理"""
        self.archive_manager.archive_old_data()    
        
    def get_kline_1s(self, start_time: str, end_time: str, symbol: str) -> List[Dict[str, Any]]:
        """获取 1 秒 K 线数据，优先从缓存获取。"""
        cached_data = self.data_cache.get_cached_data(symbol, "1s", start_time, end_time)

        if cached_data:
            return cached_data

        # 若缓存中无数据，则从数据库查询
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, price, volume, turnover
            FROM quotes
            WHERE timestamp BETWEEN ? AND ? AND symbol = ?
            ORDER BY timestamp ASC
        ''', (start_time, end_time, symbol))

        return [
            {"timestamp": row[0], "price": row[1], "volume": row[2], "turnover": row[3]}
            for row in cursor.fetchall()
        ]

    def get_kline_5s(self, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """基于 1s 数据生成 5s 数据"""
        data_1s = self.data_cache.get_cached_data(symbol, start_time, end_time)

        if not data_1s:
            return []

        kline_5s = []
        for i in range(0, len(data_1s), 5):
            segment = data_1s[i:i + 5]

            open_price = segment[0]["open"]
            close_price = segment[-1]["close"]
            high_price = max(item["high"] for item in segment)
            low_price = min(item["low"] for item in segment)
            volume = sum(item["volume"] for item in segment)
            turnover = sum(item["turnover"] for item in segment)

            kline_5s.append({
                "timestamp": segment[-1]["timestamp"],
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "turnover": turnover
            })

        return kline_5s

    def get_kline_1m(self, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """基于 5s 数据生成 1m 数据"""
        data_5s = self.get_kline_5s(symbol, start_time, end_time)

        if not data_5s:
            return []

        kline_1m = []
        for i in range(0, len(data_5s), 12):
            segment = data_5s[i:i + 12]

            open_price = segment[0]["open"]
            close_price = segment[-1]["close"]
            high_price = max(item["high"] for item in segment)
            low_price = min(item["low"] for item in segment)
            volume = sum(item["volume"] for item in segment)
            turnover = sum(item["turnover"] for item in segment)

            kline_1m.append({
                "timestamp": segment[-1]["timestamp"],
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "turnover": turnover
            })

        return kline_1m
