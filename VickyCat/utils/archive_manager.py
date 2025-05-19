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
        �鵵����������ݣ�
        - �鵵 `days_to_keep` ��ǰ�����ݵ��������ݿ��ļ�
        - ɾ�� `days_to_delete` ��ǰ������

        Args:
            days_to_keep (int): �鵵������������Ĭ�� 3 ��ǰ���ݣ�
            days_to_delete (int): ɾ��������������Ĭ�� 4 ��ǰ���ݣ�
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            archive_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
            delete_date = (datetime.now() - timedelta(days=days_to_delete)).strftime("%Y-%m-%d")
            archive_path = os.path.join(self.archive_dir, f"archive_{archive_date}.db")

            # ��ʼ���鵵���ݿ�
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

            # �鵵����
            cursor.execute(f'''
                ATTACH DATABASE '{archive_path}' AS archive_db;
                INSERT OR IGNORE INTO archive_db.quotes
                SELECT * FROM quotes WHERE timestamp < ?;
                DETACH DATABASE archive_db;
            ''', (archive_date,))

            # ɾ����������
            cursor.execute('''
                DELETE FROM quotes WHERE timestamp < ?
            ''', (delete_date,))

            conn.commit()
            print(f"[ArchiveManager] ���ݹ鵵��ɣ��鵵�ļ�: {archive_path}")

        except sqlite3.Error as e:
            print(f"[ArchiveManager] ���ݹ鵵ʧ��: {e}")

        finally:
            if conn:
                conn.close()
