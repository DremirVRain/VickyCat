import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import VickyCat
# ----------------------------- 实时K线图绘制 -----------------------------

def update_plot(frame):
    """更新K线图"""
    plt.cla()
    
    timestamps = [entry["timestamp"] for entry in kline_queue]
    opens = [entry["open"] for entry in kline_queue]
    highs = [entry["high"] for entry in kline_queue]
    lows = [entry["low"] for entry in kline_queue]
    closes = [entry["close"] for entry in kline_queue]

    if len(timestamps) >= 2:
        plt.plot(timestamps, closes, label="Close", color="blue")
        plt.fill_between(timestamps, lows, highs, color="gray", alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.title("Real-time 1s Kline")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid()
        plt.legend()
    else:
        print("Waiting for more data to plot...")

def start_plotting():
    """在独立线程中启动绘图，以避免阻塞主事件循环"""
    fig = plt.figure()
    ani = FuncAnimation(fig, update_plot, interval=1000, save_count=60)
    plt.show()

async def kline_plotter():
    """启动绘图任务"""
    threading.Thread(target=start_plotting, daemon=True).start()