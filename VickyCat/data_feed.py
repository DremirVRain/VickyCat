import asyncio
import sqlite3
from datetime import datetime
from decimal import Decimal
from collections import deque
from longport.openapi import QuoteContext, Config, SubType, PushQuote
from typing import Dict, Any

DB_PATH = "market_data_async.db"
symbols = ["TSLA.US", "TSDD.US"]
quote_queue = {
    "TSLA.US": deque(maxlen=1000),
    "TSDD.US": deque(maxlen=1000),
}

def init_db():
    """初始化数据库结构"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quotes (
            timestamp TEXT,
            symbol TEXT,
            price REAL,
            volume INTEGER,
            turnover REAL
        )
    ''')
    conn.commit()
    conn.close()

async def save_quote_to_db(quote_data: Dict[str, Any]):
    """保存逐笔行情数据"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 调试输出：打印要插入的数据
        print(f"Saving quote to DB: {quote_data}")

        cursor.execute('''
            INSERT INTO quotes (timestamp, symbol, price, volume, turnover)
            VALUES (?, ?, ?, ?, ?)
        ''', (quote_data['timestamp'], quote_data['symbol'], quote_data['price'], quote_data['volume'], quote_data['turnover']))

        conn.commit()  # 提交事务
        conn.close()  # 关闭数据库连接

        print(f"Data successfully saved to DB: {quote_data}")  # 调试输出

    except sqlite3.Error as e:
        print(f"Error while saving quote to DB: {e}")

def on_quote(symbol: str, quote: PushQuote):
    """行情推送处理"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quote_data = {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": float(quote.last_done),
        "volume": int(quote.current_volume) if quote.current_volume else 0,
        "turnover": float(quote.current_turnover) if quote.current_turnover else 0.0
    }    
    
    # 判断 symbol 是否在订阅列表中
    if symbol in quote_queue:
        quote_queue[symbol].append(quote_data)
        print(f"Received Quote for {symbol}: {quote_data}")
    else:
        print(f"Received quote for untracked symbol: {symbol}")

async def start_data_collection():
    """启动行情订阅"""
    config = Config.from_env()
    ctx = QuoteContext(config)
    ctx.set_on_quote(on_quote)
    ctx.subscribe(symbols, [SubType.Quote], True)

    print("开始订阅行情...")

    while True:
        # 遍历每个 symbol 的队列，逐一处理数据
        for symbol, q_queue in quote_queue.items():
            while q_queue:
                quote_data = q_queue.popleft()
                await save_quote_to_db(quote_data)
        
        # 控制异步循环频率，防止 CPU 占用过高
        await asyncio.sleep(0.1)
