import os
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from config import TRADE_SESSION
from utils.archive_manager import ArchiveManager
from utils.data_cache import DataCache
from utils.trading_time_manager import TradingTimeManager
from collections import defaultdict

DB_PATH = "market_data_async.db"

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.create_tables()

        # 初始化 ArchiveManager
        self.archive_manager = ArchiveManager()

        # 初始化缓存
        self.data_cache = DataCache()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                timestamp TEXT,
                symbol TEXT,
                sequence INTEGER,
                price REAL,
                open REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                turnover REAL,
                current_volume INTEGER,
                current_turnover REAL,
                trade_status TEXT,
                trade_session TEXT,
                PRIMARY KEY (timestamp, symbol, sequence)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tick_trades (
                timestamp TEXT,
                symbol TEXT,
                sequence INTEGER,
                price REAL,
                volume INTEGER,
                trade_type TEXT,
                direction TEXT,
                trade_session TEXT,
                PRIMARY KEY (timestamp, symbol, sequence)
            )
        ''')
        self.conn.commit()

    def get_next_sequence(self, table: str, timestamp: str, symbol: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT MAX(sequence) FROM {table} WHERE timestamp = ? AND symbol = ?
        ''', (timestamp, symbol))
        max_sequence = cursor.fetchone()[0]
        return (max_sequence + 1) if max_sequence is not None else 1

    #todo: 改为外部输入
    def is_trading_session(exchange: str = 'US') -> bool:
        manager = TradingTimeManager(exchange)
        return manager.is_trading_time()

    async def save_quotes_batch(self, quote_data_list: List[Dict[str, Any]]):
        if not quote_data_list:
            return

        try:
            cursor = self.conn.cursor()
            data_to_insert = []
            error_data = []
            sequence_map = {}

            for data in quote_data_list:
                try:
                    symbol = data["symbol"]
                    timestamp = data["timestamp"]
                    key = (timestamp, symbol)
                    if key not in sequence_map:
                        sequence_map[key] = self.get_next_sequence("quotes", timestamp, symbol)
                    else:
                        sequence_map[key] += 1

                    seq = sequence_map[key]

                    data_to_insert.append((
                        timestamp, symbol, seq,
                        data.get('price', 0.0),
                        data.get('open', 0.0),
                        data.get('high', 0.0),
                        data.get('low', 0.0),
                        data.get('volume', 0),
                        data.get('turnover', 0.0),
                        data.get('current_volume', 0),
                        data.get('current_turnover', 0.0),
                        data.get('trade_status', ''),
                        data.get('trade_session', '')
                    ))

                except KeyError as e:
                    print(f"[Warning] 数据字段缺失: {data} - {e}")
                    error_data.append((data, str(e)))

            if data_to_insert:
                cursor.executemany('''
                    INSERT INTO quotes (
                        timestamp, symbol, sequence,
                        price, open, high, low,
                        volume, turnover,
                        current_volume, current_turnover,
                        trade_status, trade_session
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data_to_insert)
                self.conn.commit()

            for data, msg in error_data:
                self.log_error_data(data, msg)

        except Exception as e:
            print(f"[Error] 批量插入quote数据失败: {e}")
            self.conn.rollback()

    async def save_trades_batch(self, trades: List[Dict[str, Any]]):
        """批量保存逐笔成交数据"""
        if not trades:
            return

        try:
            cursor = self.conn.cursor()
            data_to_insert = []
            sequence_map = {}

            for trade in trades:
                symbol = trade["symbol"]
                timestamp = trade["timestamp"]
                key = (timestamp, symbol)
                if key not in sequence_map:
                    sequence_map[key] = self.get_next_sequence("tick_trades", timestamp, symbol)
                else:
                    sequence_map[key] += 1

                seq = sequence_map[key]

                data_to_insert.append((
                    timestamp,
                    symbol,
                    seq,
                    trade.get("price", 0.0),
                    trade.get("volume", 0),
                    trade.get("trade_type", ""),
                    trade.get("direction", ""),
                    trade.get("trade_session", "")
                ))

            if data_to_insert:
                cursor.executemany('''
                    INSERT INTO tick_trades (
                        timestamp, symbol, sequence,
                        price, volume,
                        trade_type, direction, trade_session
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', data_to_insert)
                self.conn.commit()

        except Exception as e:
            print(f"[Error] 批量插入trade数据失败: {e}")
            self.conn.rollback()

    def archive_old_data(self):
        self.archive_manager.archive_old_data()

    def get_kline_by_period(self, period: str, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        raw_ticks = self.get_raw_ticks(symbol, start_time, end_time)
        return self.aggregate_kline(period, raw_ticks)

    def get_raw_ticks(self, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, price, volume, turnover FROM quotes
            WHERE timestamp BETWEEN ? AND ? AND symbol = ?
            ORDER BY timestamp ASC
        ''', (start_time, end_time, symbol))

        rows = cursor.fetchall()
        ticks = []
        for row in rows:
            ts, price, volume, turnover = row
            ticks.append({
                "timestamp": ts,
                "price": price,
                "volume": volume,
                "turnover": turnover
            })
        return ticks

    def aggregate_kline(self, period: str, ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ticks:
            return []

        period_seconds = {"1s": 1, "5s": 5, "1m": 60}.get(period)
        if not period_seconds:
            raise ValueError(f"Unsupported period: {period}")

        grouped = defaultdict(list)

        for tick in ticks:
            dt = datetime.strptime(tick["timestamp"], "%Y-%m-%d %H:%M:%S")
            anchor = dt - timedelta(seconds=(dt.second % period_seconds))
            if period == "1m":
                anchor = anchor.replace(second=0)
            key = anchor.strftime("%Y-%m-%d %H:%M:%S")
            grouped[key].append(tick)

        klines = []
        for ts in sorted(grouped.keys()):
            group = grouped[ts]
            open_price = group[0]["price"]
            close_price = group[-1]["price"]
            high_price = max(t["price"] for t in group)
            low_price = min(t["price"] for t in group)
            volume = sum(t["volume"] for t in group)
            turnover = sum(t["turnover"] for t in group)

            klines.append({
                "timestamp": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "turnover": turnover
            })

        return klines

    def get_quotes(self, symbol: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        sql = "SELECT * FROM quotes WHERE symbol = ?"
        params = [symbol]
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
        sql += " ORDER BY timestamp ASC"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(zip([column[0] for column in cursor.description], row)) for row in rows]
