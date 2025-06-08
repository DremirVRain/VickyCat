from strategy.base_strategy import BaseStrategy, BuySellStrategy, MarketContext
from strategy.strategy_signal import Signal, SignalType
from strategy.strategy_utils import *
from typing import Optional, List, Dict
from datetime import datetime

# ========================
# 单K线反转形态策略
# ========================

class SingleBarReversalPattern(BuySellStrategy):
    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        shadow_type: str = "lower",  # 'lower' for hammer/hanging man, 'upper' for shooting star/inverted hammer
        min_body_ratio=0.5,
        wick_ratio=2.0,
        pattern_window=3,
        signal_strength_threshold=0.6,
        max_strength=5.0,
    ):
        super().__init__(symbol)
        self.signal_type = signal_type
        self.shadow_type = shadow_type
        self.min_body_ratio = min_body_ratio
        self.wick_ratio = wick_ratio
        self.pattern_window = pattern_window
        self.signal_strength_threshold = signal_strength_threshold
        self.max_strength = max_strength

    def required_candles(self) -> int:
        return self.pattern_window

    def is_pattern(self, klines: List[dict], context: MarketContext) -> Optional[Signal]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        ls = lower_shadow(k)
        us = upper_shadow(k)

        if r == 0 or b == 0:
            return None

        is_bull = is_bullish(k)
        is_bear = is_bearish(k)

        # 统一从 context 获取趋势，避免重复计算
        if self.signal_type == SignalType.BUY and not (context.is_uptrend or context.trend_strength > 0.2):
            return None
        if self.signal_type == SignalType.SELL and not (not context.is_uptrend and context.trend_strength < -0.2):
            return None

        # 判断影线形态是否成立
        if self.shadow_type == "lower":
            wick_ok = ls > self.wick_ratio * b and us < 0.3 * r
            wick_ratio_val = ls / b
        elif self.shadow_type == "upper":
            wick_ok = us > self.wick_ratio * b and ls < 0.3 * r
            wick_ratio_val = us / b
        else:
            return None

        if not wick_ok:
            return None

        if (
            b / r >= self.min_body_ratio and
            (
                (self.signal_type == SignalType.BUY and is_bull) or
                (self.signal_type == SignalType.SELL and is_bear)
            )
        ):
            strength = min(wick_ratio_val, self.max_strength)
            if strength < self.signal_strength_threshold:
                return None

            if self.debug:
                print(f"✅ 匹配 {self.signal_type.name}: {k}")
                print(f"[{k['timestamp']}] 检测 SingleBarReversalPattern")
                print(f"  body = {b:.4f}, range = {r:.4f}, body/range = {b/r:.4f}")
                print(f"  lower_shadow = {ls:.4f}, upper_shadow = {us:.4f}")
                print(f"  shadow_type = {self.shadow_type}, wick_ratio_val = {wick_ratio_val:.2f}")
                print(f"  is_bull = {is_bull}, is_bear = {is_bear}")
                print(f"  strength = {strength:.2f}")

            return Signal(
                signal_type=self.signal_type,
                strategy_name=self.__class__.__name__,
                strength=strength
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if self.window is None or len(self.window) < self.required_candles():
            return None
        if context is None:
            # 无context时无法做趋势判断，直接返回None
            return None
        return self.is_pattern(self.window, context)



class HammerPattern(SingleBarReversalPattern):
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.BUY, shadow_type="lower", **kwargs)


class HangingManPattern(SingleBarReversalPattern):
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.SELL, shadow_type="lower", **kwargs)


class InvertedHammerPattern(SingleBarReversalPattern):
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.BUY, shadow_type="upper", **kwargs)


class ShootingStarPattern(SingleBarReversalPattern):
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.SELL, shadow_type="upper", **kwargs)


# ========================
# 双K线反转形态
# ========================
# ========================
# 双K线反转形态
# ========================
class TwoBarReversalPattern(BuySellStrategy):
    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        require_trend: Optional[str] = None,  # "up", "down", or None
        strength: float = 1.0,
        **kwargs
    ):
        super().__init__(symbol)
        self.signal_type = signal_type
        self.require_trend = require_trend
        self.strength = strength
        self.params = kwargs

    def required_candles(self) -> int:
        return 2

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        """需子类实现：判断是否为某种形态"""
        raise NotImplementedError

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or not context.recent_klines or len(context.recent_klines) < 2:
            return None

        prev, curr = context.recent_klines[-2], context.recent_klines[-1]

        # 趋势过滤
        if self.require_trend:
            if self.require_trend == "up" and not context.is_uptrend:
                return None
            if self.require_trend == "down" and context.is_uptrend:
                return None

        if self.is_valid_pattern(prev, curr):
            return Signal(
                symbol=self.symbol,
                signal_type=self.signal_type,
                strategy_name=self.__class__.__name__,
                strength=self.strength * (context.trend_strength or 1.0),
                reason=f"{self.__class__.__name__} after {self.require_trend or 'any'} trend"
            )
        return None


class BullishEngulfingPattern(TwoBarReversalPattern):
    def __init__(self, symbol: str, open_close_overlap_ratio=0.0, **kwargs):
        super().__init__(symbol, SignalType.BUY, require_trend="down",
                         open_close_overlap_ratio=open_close_overlap_ratio, **kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        ratio = self.params.get("open_close_overlap_ratio", 0.0)
        return is_bearish(prev) and is_bullish(curr) and \
               curr["open"] < prev["close"] * (1 + ratio) and \
               curr["close"] > prev["open"] * (1 - ratio)


class BearishEngulfingPattern(TwoBarReversalPattern):
    def __init__(self, symbol: str, open_close_overlap_ratio=0.0, **kwargs):
        super().__init__(symbol, SignalType.SELL, require_trend="up",
                         open_close_overlap_ratio=open_close_overlap_ratio, **kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        ratio = self.params.get("open_close_overlap_ratio", 0.0)
        return is_bullish(prev) and is_bearish(curr) and \
               curr["open"] > prev["close"] * (1 - ratio) and \
               curr["close"] < prev["open"] * (1 + ratio)


class PiercingLinePattern(TwoBarReversalPattern):
    def __init__(self, symbol: str, min_close_above_midpoint_ratio=0.5, **kwargs):
        super().__init__(symbol, SignalType.BUY, require_trend="down",
                         min_close_above_midpoint_ratio=min_close_above_midpoint_ratio, **kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        midpoint = (prev["open"] + prev["close"]) / 2
        ratio = self.params.get("min_close_above_midpoint_ratio", 0.5)
        return is_bearish(prev) and is_bullish(curr) and \
               curr["open"] < prev["low"] and \
               curr["close"] > midpoint + ratio * abs(prev["close"] - prev["open"])


class DarkCloudCoverPattern(TwoBarReversalPattern):
    def __init__(self, symbol: str, max_close_below_midpoint_ratio=0.5, **kwargs):
        super().__init__(symbol, SignalType.SELL, require_trend="up",
                         max_close_below_midpoint_ratio=max_close_below_midpoint_ratio, **kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        midpoint = (prev["open"] + prev["close"]) / 2
        ratio = self.params.get("max_close_below_midpoint_ratio", 0.5)
        return is_bullish(prev) and is_bearish(curr) and \
               curr["open"] > prev["high"] and \
               curr["close"] < midpoint - ratio * abs(prev["close"] - prev["open"])


# ========================
# 三K线反转形态
# ========================
class ThreeBarReversalPattern(BuySellStrategy):
    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        require_trend: Optional[str] = None,
        strength: float = 1.0,
        **kwargs
    ):
        super().__init__(symbol)
        self.signal_type = signal_type
        self.require_trend = require_trend  # 'up' or 'down'
        self.strength = strength
        self.params = kwargs

    def required_candles(self) -> int:
        return 3

    def is_valid_pattern(self, a: dict, b: dict, c: dict) -> bool:
        """由子类实现形态判断逻辑"""
        raise NotImplementedError

    def generate_signal(self, klines: List[dict]) -> Optional[Signal]:
        if len(klines) < 3:
            return None
        a, b, c = klines[-3:]

        # 趋势过滤（仅用于前2根）
        if self.require_trend:
            trend = detect_trend(klines[:-1])
            if not trend or trend.direction != self.require_trend:
                return None

        if self.is_valid_pattern(a, b, c):
            return Signal(
                symbol=self.symbol,
                signal_type=self.signal_type,
                strategy_name=self.__class__.__name__,
                strength=self.strength,
                reason=f"{self.__class__.__name__} after {self.require_trend or 'any'} trend"
            )
        return None

class MorningStarPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, max_mid_body_ratio=0.3, min_third_close_ratio=0.5):
        super().__init__(symbol)
        self.max_mid_body_ratio = max_mid_body_ratio  # 中间K实体最大比例（十字星或小实体）
        self.min_third_close_ratio = min_third_close_ratio  # 第三根K线收盘必须突破前K实体一半以上比例

    def required_candles(self) -> int:
        return 3

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        a, b, c = klines[-3:]
        if is_bearish(a) and body(b) <= self.max_mid_body_ratio * body(a) and is_bullish(c):
            midpoint = (a["open"] + a["close"]) / 2
            if c["close"] > midpoint + self.min_third_close_ratio * abs(a["close"] - a["open"]):
                return self.signal_type
        return None


class EveningStarPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, max_mid_body_ratio=0.3, max_third_close_ratio=0.5):
        super().__init__(symbol)
        self.max_mid_body_ratio = max_mid_body_ratio
        self.max_third_close_ratio = max_third_close_ratio

    def required_candles(self) -> int:
        return 3

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        a, b, c = klines[-3:]
        if is_bullish(a) and body(b) <= self.max_mid_body_ratio * body(a) and is_bearish(c):
            midpoint = (a["open"] + a["close"]) / 2
            if c["close"] < midpoint - self.max_third_close_ratio * abs(a["close"] - a["open"]):
                return self.signal_type
        return None


class ThreeWhiteSoldiersPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, min_close_increase_ratio=0.01):
        super().__init__(symbol)
        self.min_close_increase_ratio = min_close_increase_ratio  # 连续三根阳线收盘价必须至少上涨比例

    def required_candles(self) -> int:
        return 3

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        a, b, c = klines[-3:]
        if all(is_bullish(k) for k in [a, b, c]):
            if (b["close"] - a["close"]) / a["close"] >= self.min_close_increase_ratio and \
               (c["close"] - b["close"]) / b["close"] >= self.min_close_increase_ratio:
                return self.signal_type
        return None


class ThreeBlackCrowsPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, min_close_decrease_ratio=0.01):
        super().__init__(symbol)
        self.min_close_decrease_ratio = min_close_decrease_ratio

    def required_candles(self) -> int:
        return 3

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        a, b, c = klines[-3:]
        if all(is_bearish(k) for k in [a, b, c]):
            if (a["close"] - b["close"]) / a["close"] >= self.min_close_decrease_ratio and \
               (b["close"] - c["close"]) / b["close"] >= self.min_close_decrease_ratio:
                return self.signal_type
        return None

# ========================
# 持续形态
# ========================

class RisingThreePattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def required_candles(self) -> int:
        return 5

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-5:]
        if not (is_bullish(k[0]) and is_bullish(k[4])):
            return None
        if any(candle_range(k[i]) >= candle_range(k[0]) for i in range(1, 4)):
            return None
        if not all(k[i]["close"] < k[0]["close"] and k[i]["open"] > k[4]["open"] for i in range(1, 4)):
            return None
        if k[4]["close"] > k[0]["close"]:
            return self.signal_type
        return None

class FallingThreePattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def required_candles(self) -> int:
        return 5

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-5:]
        if not (is_bearish(k[0]) and is_bearish(k[4])):
            return None
        if any(candle_range(k[i]) >= candle_range(k[0]) for i in range(1, 4)):
            return None
        if not all(k[i]["close"] > k[0]["close"] and k[i]["open"] < k[4]["open"] for i in range(1, 4)):
            return None
        if k[4]["close"] < k[0]["close"]:
            return self.signal_type
        return None

# ========================
# 十字星
# ========================

class DojiPattern(BuySellStrategy):
    def __init__(
        self,
        symbol: str,
        doji_threshold: float = 0.1,
        signal_strength: float = 0.5
    ):
        super().__init__(symbol)
        self.doji_threshold = doji_threshold
        self.signal_strength = signal_strength

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[Signal]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None

        if is_doji(k, self.doji_threshold):
            if self.debug:
                print(f"[{k['timestamp']}] 检测 DojiPattern")
            return Signal(
                signal_type=SignalType.NEUTRAL,
                strategy_name=self.__class__.__name__,
                strength=self.signal_strength
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if self.window is None or len(self.window) < self.required_candles():
            return None
        return self.is_pattern(self.window)



class DragonflyDojiPattern(BuySellStrategy):
    def __init__(
        self,
        symbol: str,
        min_lower_shadow_ratio: float = 0.6,
        doji_threshold: float = 0.1,
        signal_strength_base: float = 1.0,
        signal_strength_cap: float = 5.0,
        trend_filter_enabled: bool = True,
        trend_window: int = 3,
    ):
        super().__init__(symbol)
        self.min_lower_shadow_ratio = min_lower_shadow_ratio
        self.doji_threshold = doji_threshold
        self.signal_strength_base = signal_strength_base
        self.signal_strength_cap = signal_strength_cap
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_window = trend_window

    def required_candles(self) -> int:
        return self.trend_window

    def is_pattern(self, klines: List[dict]) -> Optional[Signal]:
        k = klines[-1]
        r = candle_range(k)
        ls = lower_shadow(k)

        if r == 0:
            return None

        if not is_doji(k, self.doji_threshold):
            return None

        if ls / r < self.min_lower_shadow_ratio:
            return None

        if self.trend_filter_enabled and not is_downtrend(klines, self.trend_window):
            return None

        strength = min(ls / r * self.signal_strength_base, self.signal_strength_cap)
        if self.debug:
            print(f"[{k['timestamp']}] 检测 DragonflyDojiPattern")
            print(f"  lower_shadow = {ls:.4f}, range = {r:.4f}, strength = {strength:.2f}")
        return Signal(
            signal_type=SignalType.BUY,
            strategy_name=self.__class__.__name__,
            strength=strength
        )

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if self.window is None or len(self.window) < self.required_candles():
            return None
        return self.is_pattern(self.window)



class GravestoneDojiPattern(BuySellStrategy):
    def __init__(
        self,
        symbol: str,
        min_upper_shadow_ratio: float = 0.6,
        doji_threshold: float = 0.1,
        signal_strength_base: float = 1.0,
        signal_strength_cap: float = 5.0,
        trend_filter_enabled: bool = True,
        trend_window: int = 3,
    ):
        super().__init__(symbol)
        self.min_upper_shadow_ratio = min_upper_shadow_ratio
        self.doji_threshold = doji_threshold
        self.signal_strength_base = signal_strength_base
        self.signal_strength_cap = signal_strength_cap
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_window = trend_window

    def required_candles(self) -> int:
        return self.trend_window

    def is_pattern(self, klines: List[dict]) -> Optional[Signal]:
        k = klines[-1]
        r = candle_range(k)
        us = upper_shadow(k)
        b = body(k)

        if r == 0:
            return None

        if not is_doji(k, self.doji_threshold):
            return None

        if us / r < self.min_upper_shadow_ratio:
            return None

        if self.trend_filter_enabled and not is_uptrend(klines, self.trend_window):
            return None

        strength = min(us / r * self.signal_strength_base, self.signal_strength_cap)
        if self.debug:
            print(f"[{k['timestamp']}] 检测 GravestoneDojiPattern")
            print(f"  upper_shadow = {us:.4f}, range = {r:.4f}, strength = {strength:.2f}")
        return Signal(
            signal_type=SignalType.SELL,
            strategy_name=self.__class__.__name__,
            strength=strength
        )

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if self.window is None or len(self.window) < self.required_candles():
            return None
        return self.is_pattern(self.window)


# ========================
# 额外形态
# ========================

class BullishHaramiPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, max_body_ratio=0.5):
        super().__init__(symbol)
        self.max_body_ratio = max_body_ratio  # 第二根K线实体最大相对于第一根的比例

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])
        if is_bearish(prev) and is_bullish(curr):
            if curr_body <= self.max_body_ratio * prev_body:
                if curr["open"] > prev["close"] and curr["close"] < prev["open"]:
                    return self.signal_type
        return None

class BearishHaramiPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, max_body_ratio=0.5):
        super().__init__(symbol)
        self.max_body_ratio = max_body_ratio

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])
        if is_bullish(prev) and is_bearish(curr):
            if curr_body <= self.max_body_ratio * prev_body:
                if curr["open"] < prev["close"] and curr["close"] > prev["open"]:
                    return self.signal_type
        return None

class TweezerBottomPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, max_low_diff=0.001):
        super().__init__(symbol)
        self.max_low_diff = max_low_diff  # 两根K线最低价最大差异比例

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        low_diff = abs(prev["low"] - curr["low"]) / max(prev["low"], curr["low"])
        if low_diff <= self.max_low_diff and is_bearish(prev) and is_bullish(curr):
            return self.signal_type
        return None

class TweezerTopPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, max_high_diff=0.001):
        super().__init__(symbol)
        self.max_high_diff = max_high_diff

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        high_diff = abs(prev["high"] - curr["high"]) / max(prev["high"], curr["high"])
        if high_diff <= self.max_high_diff and is_bullish(prev) and is_bearish(curr):
            return self.signal_type
        return None
    
# ========================
# 不确定形态
# ========================

class InsideBarPattern(CandlePatternStrategy):
    signal_type = SignalType.WARNING

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: list) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        # 当前K线最高价不超过前K线最高价，最低价不低于前K线最低价
        if curr["high"] <= prev["high"] and curr["low"] >= prev["low"]:
            return self.signal_type
        return None

class MarubozuPattern(CandlePatternStrategy):

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: list) -> Optional[SignalType]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None

        body_len = abs(k["close"] - k["open"])
        upper_shadow_len = k["high"] - max(k["close"], k["open"])
        lower_shadow_len = min(k["close"], k["open"]) - k["low"]

        # 实体占比至少95%
        body_ratio = body_len / r
        upper_shadow_ratio = upper_shadow_len / r
        lower_shadow_ratio = lower_shadow_len / r

        if body_ratio >= 0.95 and upper_shadow_ratio <= 0.05 and lower_shadow_ratio <= 0.05:
            if is_bullish(k):
                return SignalType.BUY
            elif is_bearish(k):
                return SignalType.SELL
        return None

class SpinningTopPattern(CandlePatternStrategy):
    signal_type = SignalType.WARNING

    def __init__(self, symbol: str, max_body_ratio=0.3, min_shadow_ratio=0.3):
        super().__init__(symbol)
        self.max_body_ratio = max_body_ratio
        self.min_shadow_ratio = min_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: list) -> Optional[SignalType]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None

        body_len = abs(k["close"] - k["open"])
        upper_shadow_len = k["high"] - max(k["close"], k["open"])
        lower_shadow_len = min(k["close"], k["open"]) - k["low"]

        body_ratio = body_len / r
        upper_shadow_ratio = upper_shadow_len / r
        lower_shadow_ratio = lower_shadow_len / r

        if body_ratio <= self.max_body_ratio and upper_shadow_ratio >= self.min_shadow_ratio and lower_shadow_ratio >= self.min_shadow_ratio:
            return self.signal_type
        return None

class ThreeLineStrikePattern(CandlePatternStrategy):
    def required_candles(self) -> int:
        return 4

    def is_pattern(self, klines: list) -> Optional[SignalType]:
        k1, k2, k3, k4 = klines[-4], klines[-3], klines[-2], klines[-1]

        # 判断前三根是否同色（都多或都空）
        bullish_three = all(is_bullish(k) for k in [k1, k2, k3])
        bearish_three = all(is_bearish(k) for k in [k1, k2, k3])
        if not (bullish_three or bearish_three):
            return None

        # 第四根是否反向且大实体覆盖前三根开收盘
        def body_range(k):
            return min(k["open"], k["close"]), max(k["open"], k["close"])

        k4_body_min, k4_body_max = body_range(k4)
        k123_open_close = [k["open"] for k in [k1, k2, k3]] + [k["close"] for k in [k1, k2, k3]]
        k123_min = min(k123_open_close)
        k123_max = max(k123_open_close)

        if bullish_three and is_bearish(k4):
            if k4_body_min <= k123_min and k4_body_max >= k123_max:
                return SignalType.SELL

        if bearish_three and is_bullish(k4):
            if k4_body_min <= k123_min and k4_body_max >= k123_max:
                return SignalType.BUY

        return None
