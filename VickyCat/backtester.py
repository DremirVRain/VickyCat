# backtester.py

from collections import deque
from strategy_manager import StrategyManager
from strategy.strategy_signal import Signal
from strategy.candle_pattern_strategy import *
from database import DatabaseManager
from datetime import datetime, timedelta
import time
from utils.time_util import convert_to_eastern

@dataclass
class BacktestSignalEvent:
    timestamp: str
    symbol: str
    signal: Signal
    kline_snapshot: List[Dict[str, Any]]
    future_klines: List[Dict[str, Any]] = field(default_factory=list)
    max_profit: Optional[float] = None
    max_drawdown: Optional[float] = None
    exit_reason: Optional[str] = None

    def analyze_future(self, lookahead: int = 5):
        entry_price = self.kline_snapshot[-1]["close"]
        highs = [k["high"] for k in self.future_klines[:lookahead]]
        lows = [k["low"] for k in self.future_klines[:lookahead]]
        if not highs or not lows:
            return

        if self.signal.signal_type == "buy":
            self.max_profit = max(h - entry_price for h in highs)
            self.max_drawdown = min(l - entry_price for l in lows)
        elif self.signal.signal_type == "sell":
            self.max_profit = max(entry_price - l for l in lows)
            self.max_drawdown = min(entry_price - h for h in highs)


class Backtester:
    def __init__(self, db_path: str, symbols: list):
        self.db_manager = DatabaseManager()
        self.symbols = symbols
        self.strategy_manager = StrategyManager(self.db_manager)
        self.results = []

        self.kline_windows = {symbol: deque(maxlen=20) for symbol in symbols}
        self.pending_evals = {symbol: [] for symbol in symbols}

    def setup_strategies(self):
        for symbol in self.symbols:
            self.strategy_manager.register_strategy(symbol, HammerPattern(symbol))
            self.strategy_manager.register_strategy(symbol, HangingManPattern(symbol))
            self.strategy_manager.register_strategy(symbol, InvertedHammerPattern(symbol))
            self.strategy_manager.register_strategy(symbol, ShootingStarPattern(symbol))

        self.strategy_manager.set_signal_callback(self.on_signal)

    def on_signal(self, symbol: str, signal: Signal, kline_index: int):
        print(f"[{convert_to_eastern(signal.timestamp)}] {symbol} 触发信号: {signal.signal_type}")
        self.results.append((symbol, signal))
        self.pending_evals[symbol].append({
            'signal': signal,
            'index': kline_index
        })

    def fill_missing_minutes(self, klines: list) -> list:
        """自动填充跳空分钟蜡烛"""
        if not klines:
            return []

        filled = []
        expected_time = datetime.strptime(klines[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
        for kline in klines:
            actual_time = datetime.strptime(kline['timestamp'], "%Y-%m-%d %H:%M:%S")
            while expected_time < actual_time:
                filled.append({
                    'timestamp': expected_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'open': kline['open'],
                    'high': kline['open'],
                    'low': kline['open'],
                    'close': kline['open'],
                    'volume': 0,
                    'turnover': 0.0
                })
                expected_time += timedelta(minutes=1)
            filled.append(kline)
            expected_time = actual_time + timedelta(minutes=1)

        return filled

    def run(self, start_date: str = None, end_date: str = None):
        self.setup_strategies()

        for symbol in self.symbols:
            print(f"🚀 开始回测 {symbol} 数据")
            klines = self.db_manager.get_kline_by_period("1m", symbol, start_date, end_date)
            klines = self.fill_missing_minutes(klines)
            print(f"共加载（含补全）{len(klines)} 根分钟K线")

            pending = self.pending_evals[symbol]

            for idx, kline in enumerate(klines):
                self.strategy_manager.on_kline(symbol, kline, idx)
                time.sleep(0.001)

            for eval_info in pending:
                i = eval_info['index']
                if i + 7 < len(klines):
                    entry_price = eval_info['signal'].price
                    close_price = klines[i + 7]['close']
                    pnl = close_price - entry_price if eval_info['signal'].signal_type == SignalType.BUY else entry_price - close_price
                    eval_info['pnl'] = pnl
                    print(f"📊 [{convert_to_eastern(klines[i + 7]['timestamp'])}] {symbol} 第7根评估: 收盘={close_price:.2f}, 盈亏={pnl:.2f}")
                else:
                    print(f"⚠️ {symbol} 信号在末尾，无法评估")

        all_pnls = [e['pnl'] for s in self.symbols for e in self.pending_evals[s] if 'pnl' in e]
        total = sum(all_pnls)
        win = sum(1 for p in all_pnls if p > 0)
        loss = sum(1 for p in all_pnls if p <= 0)
        win_rate = win / len(all_pnls) if all_pnls else 0
        avg_win = sum(p for p in all_pnls if p > 0) / win if win > 0 else 0
        avg_loss = -sum(p for p in all_pnls if p <= 0) / loss if loss > 0 else 0
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        print(f"\n✅ 回测完成，共触发 {len(self.results)} 个信号")
        print(f"📈 胜率: {win_rate:.2%}, 盈亏比: {rr_ratio:.2f}, 总盈亏: {total:.2f}")


if __name__ == "__main__":
    symbols = ["TSLA.US", "TSDD.US"]
    db_path = "minute_data.db"
    start_date = "2025-05-27 21:30:00"
    end_date = "2025-05-28 05:30:00"

    backtester = Backtester(db_path, symbols)
    backtester.run(start_date, end_date)
