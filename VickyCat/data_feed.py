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
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    """保存逐笔行情数据，带 sequence 字段"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 获取当前秒内的最大 sequence 值
        cursor.execute('''
            SELECT MAX(sequence) FROM quotes 
            WHERE timestamp = ? AND symbol = ?
        ''', (quote_data['timestamp'], quote_data['symbol']))
        
        max_sequence = cursor.fetchone()[0]
        sequence = (max_sequence + 1) if max_sequence is not None else 1

        # 插入数据
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

        conn.commit()

        print(f"Data saved to DB: {quote_data}, sequence: {sequence}")

    except sqlite3.Error as e:
        print(f"Error while saving quote to DB: {e}")

    finally:
        if conn:
            conn.close()

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
        try:
            for symbol, q_queue in quote_queue.items():
                while q_queue:
                    quote_data = q_queue.popleft()
                    await save_quote_to_db(quote_data)
        except Exception as e:
            print(f"Error in start_data_collection loop: {e}")
        await asyncio.sleep(0.1)
