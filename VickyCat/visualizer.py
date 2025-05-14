import asyncio
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.animation import FuncAnimation

DB_PATH = "market_data_async.db"

# K线数据缓存队列
kline_cache = []

def fetch_latest_kline_data(limit=60):
    """从数据库获取最新的K线数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, open, high, low, close, volume, turnover FROM kline_1s
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    data = cursor.fetchall()
    conn.close()
    # 最新数据在前，倒序排列
    return data[::-1]

def update_plot(frame):
    """更新K线图"""
    plt.cla()
    data = fetch_latest_kline_data()

    if not data:
        print("暂无K线数据")
        return

    timestamps = [entry[0] for entry in data]
    opens = [entry[1] for entry in data]
    highs = [entry[2] for entry in data]
    lows = [entry[3] for entry in data]
    closes = [entry[4] for entry in data]

    # 绘制蜡烛图
    plt.plot(timestamps, closes, label="Close", color="blue")
    plt.fill_between(timestamps, lows, highs, color="gray", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Real-time 1s Kline")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid()
    plt.legend()

async def start_visualization():
    """启动实时可视化"""
    fig = plt.figure()
    ani = FuncAnimation(fig, update_plot, interval=1000, save_count=60)
    plt.show()
