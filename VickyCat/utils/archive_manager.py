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
        """
        归档并清理旧数据：
        - 归档 `days_to_keep` 天前的数据到单独数据库文件
        - 删除 `days_to_delete` 天前的数据

        Args:
            days_to_keep (int): 归档的数据天数（默认 3 天前数据）
            days_to_delete (int): 删除的数据天数（默认 4 天前数据）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            archive_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
            delete_date = (datetime.now() - timedelta(days=days_to_delete)).strftime("%Y-%m-%d")
            archive_path = os.path.join(self.archive_dir, f"archive_{archive_date}.db")

            # 初始化归档数据库
            if not os.path.exists(archive_path):
                with sqlite3.connect(archive_path) as archive_conn:
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

            # 归档数据
            cursor.execute(f'''
                ATTACH DATABASE '{archive_path}' AS archive_db;
                INSERT OR IGNORE INTO archive_db.quotes
                SELECT * FROM quotes WHERE timestamp < ?;
                DETACH DATABASE archive_db;
            ''', (archive_date,))

            # 删除超期数据
            cursor.execute('''
                DELETE FROM quotes WHERE timestamp < ?
            ''', (delete_date,))

            conn.commit()
            print(f"[ArchiveManager] 数据归档完成，归档文件: {archive_path}")

        except sqlite3.Error as e:
            print(f"[ArchiveManager] 数据归档失败: {e}")

        finally:
            if conn:
                conn.close()
