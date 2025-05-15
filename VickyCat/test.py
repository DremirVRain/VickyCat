import asyncio
import datetime
from decimal import Decimal
from typing import Any, Dict
import sqlite3
from trade import Trade

import inspect
from longport.openapi import TradeContext, Config, OrderType, OrderSide, TimeInForceType, OrderStatus, Market
import sqlite3

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders_test (
            order_id TEXT PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            price REAL,
            quantity REAL,
            status TEXT,
            executed_price REAL,
            executed_quantity REAL,
            remark TEXT
        )
    ''')
    conn.commit()
    conn.close()

from typing import Optional
# 你之前定义的数据库路径
DB_PATH = "your_orders.db"

# 将 Decimal 转为 float（方便存数据库）
def dec_to_float(d: Any) -> Any:
    if isinstance(d, Decimal):
        return float(d)
    return d

# Enum 转字符串（OrderSide, OrderStatus等）
def enum_to_str(e: Any) -> str:
    return str(e) if e is not None else ""

# 时间转字符串
def datetime_to_str(dt: Any) -> str:
    if isinstance(dt, datetime.datetime):
        return dt.isoformat()
    if isinstance(dt, datetime.date):
        return dt.isoformat()
    return ""

async def save_order_to_db(order_data: Dict[str, Any]):
    """保存订单数据到数据库 orders_test 表"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO orders_test 
            (order_id, timestamp, symbol, action, price, quantity, status, executed_price, executed_quantity, remark)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order_data['order_id'],
            datetime_to_str(order_data['timestamp']),
            order_data['symbol'],
            enum_to_str(order_data['action']),
            dec_to_float(order_data['price']),
            dec_to_float(order_data['quantity']),
            enum_to_str(order_data['status']),
            dec_to_float(order_data['executed_price']),
            dec_to_float(order_data['executed_quantity']),
            order_data['remark'],
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error while saving order to DB: {e}")
    finally:
        if conn:
            conn.close()

# 下面是调用示例函数，调用 submit_order 并保存结果
async def submit_and_save_order(trade_obj, symbol: str, submitted_quantity: Decimal, price: Decimal, remark: str = "", outside_rth: Optional[bool] = None):
    try:
        result = trade_obj.submit_order(
            symbol=symbol,
            order_type=OrderType.LO,
            side=OrderSide.Buy,
            submitted_quantity=submitted_quantity,
            time_in_force=TimeInForceType.GoodTilCanceled,
            submitted_price=price,
            outside_rth=outside_rth,
            remark=remark,
        )

        # 检查是否返回了 order_id
        if not hasattr(result, "order_id"):
            print("Order submission failed: No order_id in response")
            return

        order_id = result.order_id

        order_data = {
            "order_id": order_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "symbol": symbol,
            "action": "Buy",
            "price": float(price),
            "quantity": float(submitted_quantity),
            "status": "SUBMITTED",
            "executed_price": 0.0,
            "executed_quantity": 0.0,
            "remark": remark,
        }

        await save_order_to_db(order_data)
        print(f"Order {order_id} saved successfully.")

    except Exception as e:
        print(f"Order submission failed: {e}")

# 你可以这样运行（示例）
async def main():
    initialize_db()
    trade = Trade()
    await submit_and_save_order(trade, "TSLA.US", Decimal("1"), Decimal("700.0"), "test order")

asyncio.run(main())