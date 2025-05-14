import asyncio
import sqlite3
from datetime import datetime
from longport.openapi import QuoteContext, Config, SubType, PushQuote
from collections import deque
from aiohttp import ClientSession
from decimal import Decimal
from typing import Optional, Dict

DB_PATH = "market_data_async.db"
symbols = ["TSLA.US"]
quote_queue = deque(maxlen=1000)  # 异步数据队列
kline_queue = deque(maxlen=1000)

def check_table_structure():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(kline_1s)")
    columns = cursor.fetchall()
    for column in columns:
        print(column)
    conn.close()

def update_kline_1s_structure():
    """更新 kline_1s 表结构，新增 volume 和 turnover 字段"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 检查现有字段，避免重复添加
    cursor.execute("PRAGMA table_info(kline_1s)")
    columns = [col[1] for col in cursor.fetchall()]

    # 新增 volume 字段，类型为 INTEGER，默认值为 0
    if "volume" not in columns:
        cursor.execute("ALTER TABLE kline_1s ADD COLUMN volume INTEGER DEFAULT 0")
        print("新增字段 volume")

    # 新增 turnover 字段，类型为 REAL，默认值为 0.0
    if "turnover" not in columns:
        cursor.execute("ALTER TABLE kline_1s ADD COLUMN turnover REAL DEFAULT 0.0")
        print("新增字段 turnover")

    # 新增 volume_diff 字段，类型为 INTEGER，默认值为 0
    if "volume_diff" not in columns:
        cursor.execute("ALTER TABLE kline_1s ADD COLUMN volume_diff INTEGER DEFAULT 0")
        print("新增字段 volume_diff")

    # 新增 turnover_diff 字段，类型为 REAL，默认值为 0.0
    if "turnover_diff" not in columns:
        cursor.execute("ALTER TABLE kline_1s ADD COLUMN turnover_diff REAL DEFAULT 0.0")
        print("新增字段 turnover_diff")

    conn.commit()
    conn.close()

def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quotes (
            timestamp TEXT,
            symbol TEXT,
            price REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kline_1s (
            timestamp TEXT PRIMARY KEY,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL
        )
    ''')
    conn.commit()
    conn.close()

async def save_quote_to_db(symbol: str, price: float):
    """异步保存逐笔数据"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO quotes (timestamp, symbol, price)
        VALUES (?, ?, ?)
    ''', (timestamp, symbol, price))
    conn.commit()
    conn.close()

async def fetch_previous_data():
    """获取上一秒的成交量和成交金额"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT volume, turnover FROM kline_1s 
        ORDER BY timestamp DESC LIMIT 1
    """)
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0], result[1]
    else:
        # 若无前一秒数据，返回 (0, 0)
        return 0, 0

    if result:
        return result[0], result[1]
    else:
        # 若无前一秒数据，设为0
        return 0, 0

async def save_kline_to_db(kline):
    """异步保存1秒K线数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 确保数值数据为 float 类型
    timestamp = kline["timestamp"]
    symbol = kline["symbol"]
    open_price = float(kline["open"])
    high_price = float(kline["high"])
    low_price = float(kline["low"])
    close_price = float(kline["close"])
    current_volume = float(kline.get("volume", 0))
    current_turnover = float(kline.get("turnover", 0))  # Default to 0 if 'volume' is missing
    
    # 获取上一秒成交量和成交金额
    previous_volume, previous_turnover = await fetch_previous_data()

    # 计算差分
    volume_diff = current_volume - previous_volume
    turnover_diff = current_turnover - previous_turnover
    
    # 插入数据
    try:
        cursor.execute('''
            INSERT INTO kline_1s 
            (timestamp, symbol, open, high, low, close, volume, turnover, volume_diff, turnover_diff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, symbol, open_price, high_price, low_price, close_price, 
              current_volume, current_turnover, volume_diff, turnover_diff))
        conn.commit()
    except Exception as e:
        current_volume = float(kline.get("volume", 0))  # Default to 0 if 'volume' is missing
        print(f"数据库插入错误: {e}")
    finally:
        conn.close()
        
def create_1s_kline(symbol: str) -> Optional[Dict[str, any]]:
    """生成1秒K线数据，包含volume和turnover字段"""
    now = datetime.now()
    end_time = now.strftime("%Y-%m-%d %H:%M:%S")
    start_time = (now.timestamp() - 1)
    start_time = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")

    # 筛选当前 1 秒内的数据
    relevant_data = [
        entry for entry in list(quote_queue) 
        if start_time <= entry["timestamp"] <= end_time
    ]

    if not relevant_data:
        return None

    # 获取价格数据
    prices = [entry["price"] for entry in relevant_data]

    # 获取成交量和成交金额数据
    volumes = [entry["volume"] for entry in relevant_data]
    turnovers = [entry["turnover"] for entry in relevant_data]

    kline = {
        "timestamp": end_time,
        "symbol": symbol,
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
        "volume": sum(volumes),
        "turnover": sum(turnovers)
    }
    
    kline_queue.append(kline)

    return kline

def on_quote(symbol: str, quote: PushQuote):
    """实时行情推送处理函数，确保数据结构完整性"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 统一数据结构，若 volume 和 turnover 不存在，设为 0
    quote_data = {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": Decimal(quote.last_done),
        "volume": int(quote.current_volume),
        "turnover": float(quote.current_turnover)
    }

    # 将数据插入队列
    quote_queue.append(quote_data)
    print(f"Received Quote: {quote_data}")  # 调试输出

async def kline_producer(symbol: str):
    """异步生成1秒K线"""
    while True:
        kline = create_1s_kline(symbol)
        if kline:
            await save_kline_to_db(kline)
            print(f"Saved Kline: {kline}")  # 调试输出
        await asyncio.sleep(1)  # 每秒检查一次

async def start_quote_stream():
    """异步启动行情订阅"""
    config = Config.from_env()
    ctx = QuoteContext(config)
    ctx.set_on_quote(on_quote)
    ctx.subscribe(symbols, [SubType.Quote], True)

    print("异步订阅行情中...")

    while True:
        await asyncio.sleep(1)
    #await asyncio.Future()  # 永久运行

async def main():
    check_table_structure()
    update_kline_1s_structure()
    init_db()
    # 启动行情流和K线构建任务
    await asyncio.gather(
        start_quote_stream(),
        kline_producer(symbols[0]),
        kline_plotter()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序手动终止")