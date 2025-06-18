import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = "market_data_async.db"
ARCHIVE_DIR = "archives"

class ArchiveManager:
    def __init__(self, db_path: str = DB_PATH, archive_dir: str = ARCHIVE_DIR):
        self.db_path = db_path
        self.archive_dir = archive_dir
        os.makedirs(self.archive_dir, exist_ok=True)

    def archive_old_data(self, days_to_keep: int = 3, days_to_delete: int = 4) -> None:
        try:
            archive_date = (datetime.utcnow() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
            delete_date = (datetime.utcnow() - timedelta(days=days_to_delete)).strftime("%Y-%m-%d")
            archive_path = os.path.join(self.archive_dir, f"archive_{archive_date}.db")

            # 连接归档数据库和主数据库
            with sqlite3.connect(archive_path) as archive_conn, \
                 sqlite3.connect(self.db_path) as main_conn:

                self._ensure_archive_schema(archive_conn)

                # ==== 归档 quotes 表 ====
                quotes_data = main_conn.execute(
                    "SELECT * FROM quotes WHERE timestamp < ?", (archive_date,)
                ).fetchall()

                if quotes_data:
                    archive_conn.executemany('''
                        INSERT OR IGNORE INTO quotes (
                            timestamp, symbol, sequence,
                            price, open, high, low,
                            volume, turnover,
                            current_volume, current_turnover,
                            trade_status, trade_session
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', quotes_data)

                # ==== 归档 tick_trades 表 ====
                trades_data = main_conn.execute(
                    "SELECT * FROM tick_trades WHERE timestamp < ?", (archive_date,)
                ).fetchall()

                if trades_data:
                    archive_conn.executemany('''
                        INSERT OR IGNORE INTO tick_trades (
                            timestamp, symbol, sequence,
                            price, volume,
                            trade_type, direction, trade_session
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', trades_data)

                print(f"[ArchiveManager] 成功归档 quotes: {len(quotes_data)} 条, trades: {len(trades_data)} 条 至 {archive_path}")

            # ==== 删除主库中旧数据 ====
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM quotes WHERE timestamp < ?", (delete_date,))
                conn.execute("DELETE FROM tick_trades WHERE timestamp < ?", (delete_date,))
                conn.commit()

        except sqlite3.Error as e:
            print(f"[ArchiveManager] 数据归档失败: {e}")

    def _ensure_archive_schema(self, conn: sqlite3.Connection) -> None:
        """确保归档数据库拥有正确的表结构"""
        conn.execute('''
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
        conn.execute('''
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
        conn.commit()
