from base_strategy import BaseStrategy
from strategy_signal import Signal, SignalType
from typing import Optional

class EventDrivenStrategy(BaseStrategy):
    """事件驱动策略基类（例如财报、政策、开盘跳空）"""
    def check_event_trigger(self, kline: dict) -> Optional[Signal]:
        raise NotImplementedError

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        return self.check_event_trigger(kline)

class GapOpenEvent(EventDrivenStrategy):
    """开盘跳空策略（模拟开盘缺口）"""
    def check_event_trigger(self, kline: dict) -> Optional[Signal]:
        prev_close = kline.get("prev_close")
        open_price = kline["open"]
        if prev_close is None:
            return None

        gap_percent = abs(open_price - prev_close) / prev_close
        if gap_percent > 0.02:
            signal_type = SignalType.BUY if open_price > prev_close else SignalType.SELL
            return Signal(symbol=kline["symbol"], timestamp=kline["timestamp"],
                          signal_type=signal_type, strength=1.0,
                          strategy_name=self.__class__.__name__, metadata=kline)
        return None

