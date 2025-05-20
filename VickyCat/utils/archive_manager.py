import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = "market_data_async.db"
ARCHIVE_DIR = "archives"

class ArchiveManager:
    def __init__(self, db_path: str = DB_PATH, archive_dir: str = ARCHIVE_DIR):
        self.db_path = db_path
        self.archive_dir = archive_dir
        os.makedirs(self.archive_dir, exist_ok=True)

    def archive_old_data(self, days_to_keep: int = 3, days_to_delete: int = 4) -> None:
        try:
            archive_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
            delete_date = (datetime.now() - timedelta(days=days_to_delete)).strftime("%Y-%m-%d")
            archive_path = os.path.join(self.archive_dir, f"archive_{archive_date}.db")

            # 初始化归档库并复制数据（独立连接）
            with sqlite3.connect(archive_path) as archive_conn, \
                 sqlite3.connect(self.db_path) as main_conn:
            
                # 确保 archive 库有表结构
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

                # 获取需要归档的数据
                data = main_conn.execute(
                    "SELECT * FROM quotes WHERE timestamp < ?", (archive_date,)
                ).fetchall()

                # 写入归档数据库
                if data:
                    archive_conn.executemany('''
                        INSERT OR IGNORE INTO quotes (timestamp, symbol, sequence, price, volume, turnover)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', data)

                print(f"[ArchiveManager] 成功归档 {len(data)} 条数据至 {archive_path}")

            # 删除旧数据（另起连接）
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM quotes WHERE timestamp < ?", (delete_date,))
                conn.commit()

        except sqlite3.Error as e:
            print(f"[ArchiveManager] 数据归档失败: {e}")


