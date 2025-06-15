from strategy.base_strategy import BaseStrategy, MarketContext
from strategy.strategy_signal import Signal, SignalType
from strategy.strategy_utils import *
from typing import Optional, List, Dict
from datetime import datetime

class CandlePatternStrategy(BaseStrategy):
    strategy_category = "pattern"

    default_params = {
    }

    param_space = {
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def  generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        raise NotImplementedError

# ========================
# 单K线反转形态策略
# ========================
class SingleBarReversalPattern(CandlePatternStrategy):
    default_params = {
        "min_body_ratio": 0.5,
        "wick_ratio": 2.0,
        "pattern_window": 3,
        "signal_strength_threshold": 0.6,
        "max_strength": 5.0,
    }

    param_space = {
    "min_body_ratio": [0.3, 0.4, 0.5, 0.6],
    "wick_ratio": [1.5, 2.0, 2.5, 3.0],
    "signal_strength_threshold": [0.4, 0.5, 0.6],
    "max_strength": [3.0, 4.0, 5.0, 6.0],
    }

    def __init__(self, symbol: str, signal_type: SignalType, shadow_type: str = "lower", **kwargs):
        super().__init__(symbol)
        self.signal_type = signal_type
        self.shadow_type = shadow_type
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return self.params["pattern_window"]

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

        trend_strength = context.trend_strength()
        is_up = context.is_uptrend()

        if self.signal_type == SignalType.BUY and not (is_up or trend_strength > 0.2):
            return None
        if self.signal_type == SignalType.SELL and not (not is_up and trend_strength < -0.2):
            return None

        wick_ratio = self.params["wick_ratio"]
        min_body_ratio = self.params["min_body_ratio"]
        max_strength = self.params["max_strength"]
        signal_strength_threshold = self.params["signal_strength_threshold"]

        if self.shadow_type == "lower":
            wick_ok = ls > wick_ratio * b and us < 0.3 * r
            wick_ratio_val = ls / b
        elif self.shadow_type == "upper":
            wick_ok = us > wick_ratio * b and ls < 0.3 * r
            wick_ratio_val = us / b
        else:
            return None

        if not wick_ok:
            return None

        if (
            b / r >= min_body_ratio and
            ((self.signal_type == SignalType.BUY and is_bull) or
             (self.signal_type == SignalType.SELL and is_bear))
        ):
            strength = min(wick_ratio_val, max_strength)
            if strength < signal_strength_threshold:
                return None

            return self.build_signal(
                kline=k,
                signal_type=self.signal_type,
                strength=strength,
                metadata={
                    "reason": f"{self.__class__.__name__} with shadow_type={self.shadow_type}, trend={getattr(context, 'trend_info', 'unknown')}"
                }
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.recent_klines) < self.required_candles():
            return None
        klines = context.recent_klines[-self.required_candles():]
        return self.is_pattern(klines, context)


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
# 双K线反转形态（含信号强度计算）
# ========================
class TwoBarReversalPattern(CandlePatternStrategy):
    default_params = {
        "base_strength": 1.0,
        "max_strength": 5.0,
        "open_close_overlap_ratio": 0.0,
        "min_close_above_midpoint_ratio": 0.5,
        "max_close_below_midpoint_ratio": 0.5,
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "max_strength": [3.0, 5.0, 7.0],
        "open_close_overlap_ratio": [0.0, 0.01, 0.02],
        "min_close_above_midpoint_ratio": [0.3, 0.5, 0.7],
        "max_close_below_midpoint_ratio": [0.3, 0.5, 0.7],
    }

    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        require_trend: Optional[str] = None,
        **kwargs
    ):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.signal_type = signal_type
        self.require_trend = require_trend
        self.max_strength = self.params["max_strength"]

    def required_candles(self) -> int:
        return 2

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        raise NotImplementedError

    def compute_strength(self, prev: dict, curr: dict, context: Optional[MarketContext] = None) -> float:
        structure_strength = 1.0
        trend_strength = context.trend_strength()
        base_strength = self.params["base_strength"]
        max_strength = self.params["max_strength"]
        return min(base_strength * structure_strength * trend_strength, max_strength)

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.recent_klines) < 2:
            return None

        prev, curr = context.recent_klines[-2], context.recent_klines[-1]

        if self.require_trend == "up" and context.is_uptrend() is not True:
            return None
        if self.require_trend == "down" and context.is_uptrend() is not False:
            return None

        if self.is_valid_pattern(prev, curr):
            strength = self.compute_strength(prev, curr, context)
            if strength < 0.1:
                return None
            return self.build_signal(
                kline=curr,
                signal_type=self.signal_type,
                strength=strength,
                metadata={
                    "reason": f"{self.__class__.__name__} with trend={getattr(context, 'trend_info', 'unknown')}"
                }
            )
        return None


class BullishEngulfingPattern(TwoBarReversalPattern):
    default_params = {
        "base_strength": 1.0,
        "max_strength": 5.0,
        "open_close_overlap_ratio": 0.0
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "max_strength": [3.0, 5.0, 7.0],
        "open_close_overlap_ratio": [0.0, 0.01, 0.02]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.BUY, require_trend="down", **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        ratio = self.params.get("open_close_overlap_ratio", 0.0)
        return is_bearish(prev) and is_bullish(curr) and \
               curr["open"] < prev["close"] * (1 + ratio) and \
               curr["close"] > prev["open"] * (1 - ratio)

    def compute_strength(self, prev: dict, curr: dict, context: Optional[MarketContext] = None) -> float:
        engulf_size = abs(curr["close"] - curr["open"])
        prev_body = abs(prev["close"] - prev["open"])
        structure_strength = min(engulf_size / (prev_body + 1e-6), 2.0)
        trend_strength = context.trend_strength()
        base_strength = self.params["base_strength"]
        max_strength = self.params["max_strength"]
        return min(base_strength * structure_strength * trend_strength, max_strength)


class BearishEngulfingPattern(TwoBarReversalPattern):
    default_params = {
        "base_strength": 1.0,
        "max_strength": 5.0,
        "open_close_overlap_ratio": 0.0
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "max_strength": [3.0, 5.0, 7.0],
        "open_close_overlap_ratio": [0.0, 0.01, 0.02]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.SELL, require_trend="up", **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        ratio = self.params.get("open_close_overlap_ratio", 0.0)
        return is_bullish(prev) and is_bearish(curr) and \
               curr["open"] > prev["close"] * (1 - ratio) and \
               curr["close"] < prev["open"] * (1 + ratio)

    def compute_strength(self, prev: dict, curr: dict, context: Optional[MarketContext] = None) -> float:
        engulf_size = abs(curr["open"] - curr["close"])
        prev_body = abs(prev["close"] - prev["open"])
        structure_strength = min(engulf_size / (prev_body + 1e-6), 2.0)
        trend_strength = context.trend_strength()
        base_strength = self.params["base_strength"]
        max_strength = self.params["max_strength"]
        return min(base_strength * structure_strength * trend_strength, max_strength)


class PiercingLinePattern(TwoBarReversalPattern):
    default_params = {
        "base_strength": 1.0,
        "max_strength": 5.0,
        "min_close_above_midpoint_ratio": 0.5
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "max_strength": [3.0, 5.0, 7.0],
        "min_close_above_midpoint_ratio": [0.3, 0.5, 0.7]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.BUY, require_trend="down", **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        midpoint = (prev["open"] + prev["close"]) / 2
        ratio = self.params.get("min_close_above_midpoint_ratio", 0.5)
        return is_bearish(prev) and is_bullish(curr) and \
               curr["open"] < prev["low"] and \
               curr["close"] > midpoint + ratio * abs(prev["close"] - prev["open"])

    def compute_strength(self, prev: dict, curr: dict, context: Optional[MarketContext] = None) -> float:
        midpoint = (prev["open"] + prev["close"]) / 2
        close_penetration = max(0.0, curr["close"] - midpoint)
        body_size = abs(prev["close"] - prev["open"])
        structure_strength = min(close_penetration / (body_size + 1e-6), 2.0)
        trend_strength = context.trend_strength()
        base_strength = self.params["base_strength"]
        max_strength = self.params["max_strength"]
        return min(base_strength * structure_strength * trend_strength, max_strength)


class DarkCloudCoverPattern(TwoBarReversalPattern):
    default_params = {
        "base_strength": 1.0,
        "max_strength": 5.0,
        "max_close_below_midpoint_ratio": 0.5
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "max_strength": [3.0, 5.0, 7.0],
        "max_close_below_midpoint_ratio": [0.3, 0.5, 0.7]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, SignalType.SELL, require_trend="up", **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def is_valid_pattern(self, prev: dict, curr: dict) -> bool:
        midpoint = (prev["open"] + prev["close"]) / 2
        ratio = self.params.get("max_close_below_midpoint_ratio", 0.5)
        return is_bullish(prev) and is_bearish(curr) and \
               curr["open"] > prev["high"] and \
               curr["close"] < midpoint - ratio * abs(prev["close"] - prev["open"])

    def compute_strength(self, prev: dict, curr: dict, context: Optional[MarketContext] = None) -> float:
        midpoint = (prev["open"] + prev["close"]) / 2
        close_drop = max(0.0, midpoint - curr["close"])
        body_size = abs(prev["close"] - prev["open"])
        structure_strength = min(close_drop / (body_size + 1e-6), 2.0)
        trend_strength =context.trend_strength()
        base_strength = self.params["base_strength"]
        max_strength = self.params["max_strength"]
        return min(base_strength * structure_strength * trend_strength, max_strength)


# ========================
# 三K线反转形态
# ========================
class ThreeBarReversalPattern(CandlePatternStrategy):
    default_params = {
        "base_strength": 1.0
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5]
    }

    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        require_trend: Optional[str] = None,  # 'up' or 'down'
        **kwargs
    ):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.signal_type = signal_type
        self.require_trend = require_trend

    def required_candles(self) -> int:
        return 3

    def is_valid_pattern(self, a: dict, b: dict, c: dict) -> bool:
        """由子类实现：判断是否为有效三K反转形态"""
        raise NotImplementedError

    def calculate_strength(self, a: dict, b: dict, c: dict, context: MarketContext) -> float:
        """默认强度计算：使用三K实体强度 + 趋势力度加权"""
        body_ratio = lambda k: abs(k["close"] - k["open"]) / (k["high"] - k["low"] + 1e-6)
        avg_body = sum([body_ratio(k) for k in [a, b, c]]) / 3

        trend_weight = context.trend_strength()
        volatility_weight = context.volatility if context.volatility else 1.0
        base_strength = self.params["base_strength"]

        return base_strength * avg_body * (0.7 + 0.3 * trend_weight) * (0.8 + 0.2 * volatility_weight)

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.recent_klines) < 3:
            return None
        a, b, c = context.recent_klines[-3:]

        # 趋势过滤（使用 context 提供的信息）
        if self.require_trend:
            direction = context.trend_info.get("direction") if context.trend_info else None
            if not direction or direction != self.require_trend:
                return None

        if self.is_valid_pattern(a, b, c):
            strength = self.calculate_strength(a, b, c, context)
            return self.build_signal(
                kline=c,
                signal_type=self.signal_type,
                strength=strength,
                metadata={"reason": f"{self.__class__.__name__} after {self.require_trend or 'any'} trend"},
                expected_action="open_long" if self.signal_type == SignalType.BUY else "open_short"
            )
        return None


class MorningStarPattern(ThreeBarReversalPattern):    
    default_params = {
        "base_strength": 1.0,
        "max_mid_body_ratio": 0.3,
        "min_third_close_ratio": 0.5,
        "require_gap": True
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "max_mid_body_ratio": [0.2, 0.3, 0.4],
        "min_third_close_ratio": [0.4, 0.5, 0.6],
        "require_gap": [True, False]
    }

    def __init__(self, symbol: str,  **kwargs):
        super().__init__(symbol, signal_type=SignalType.BUY, require_trend='down', **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.max_mid_body_ratio = self.params.get("max_mid_body_ratio", 0.3)
        self.min_third_close_ratio = self.params.get("min_third_close_ratio", 0.5)
        self.require_gap = self.params.get("min_third_close_ratio",True)  # 新增是否必须跳空参数

    def is_valid_pattern(self, a: dict, b: dict, c: dict) -> bool:
        if is_bearish(a) and is_bullish(c):
            # 中间K线实体足够小
            if body(b) <= self.max_mid_body_ratio * body(a):
                # 跳空判断
                if self.require_gap:
                    if not (gap_down(b, a) and gap_up(c, b)):
                        return False
                a_mid = midpoint(a)
                required_close = a_mid + self.min_third_close_ratio * abs(a["open"] - a["close"])
                return c["close"] > required_close
        return False


class EveningStarPattern(ThreeBarReversalPattern):
    default_params = {
        "base_strength": 1.0,
        "max_mid_body_ratio": 0.3,
        "max_third_close_ratio": 0.5,
        "require_gap": True
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "max_mid_body_ratio": [0.2, 0.3, 0.4],
        "max_third_close_ratio": [0.4, 0.5, 0.6],
        "require_gap": [True, False]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, signal_type=SignalType.SELL, require_trend='up', **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.max_mid_body_ratio = self.params.get("max_mid_body_ratio", 0.3)
        self.max_third_close_ratio = self.params.get("max_third_close_ratio", 0.5)
        self.require_gap = self.params.get("require_gap", True)

    def is_valid_pattern(self, a: dict, b: dict, c: dict) -> bool:
        if is_bullish(a) and is_bearish(c):
            if body(b) <= self.max_mid_body_ratio * body(a):
                if self.require_gap:
                    if not (gap_up(b, a) and gap_down(c, b)):
                        return False
                a_mid = midpoint(a)
                required_close = a_mid - self.max_third_close_ratio * abs(a["open"] - a["close"])
                return c["close"] < required_close
        return False


class ThreeWhiteSoldiersPattern(ThreeBarReversalPattern):
    default_params = {
        "base_strength": 1.0,
        "min_close_increase_ratio": 0.01
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "min_close_increase_ratio": [0.005, 0.01, 0.02]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, signal_type=SignalType.BUY, require_trend='down', **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.min_close_increase_ratio = self.params.get("min_close_increase_ratio", 0.01)

    def is_valid_pattern(self, a: dict, b: dict, c: dict) -> bool:
        if all(is_bullish(k) for k in [a, b, c]):
            if a["close"] <= 0 or b["close"] <= 0:
                return False
            r1 = (b["close"] - a["close"]) / a["close"]
            r2 = (c["close"] - b["close"]) / b["close"]
            return r1 >= self.min_close_increase_ratio and r2 >= self.min_close_increase_ratio
        return False


class ThreeBlackCrowsPattern(ThreeBarReversalPattern):
    default_params = {
        "base_strength": 1.0,
        "min_close_decrease_ratio": 0.01
    }

    param_space = {
        "base_strength": [0.5, 1.0, 1.5],
        "min_close_decrease_ratio": [0.005, 0.01, 0.02]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, signal_type=SignalType.SELL, require_trend='up', **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.min_close_decrease_ratio = self.params.get("min_close_decrease_ratio", 0.01)

    def is_valid_pattern(self, a: dict, b: dict, c: dict) -> bool:
        if all(is_bearish(k) for k in [a, b, c]):
            if a["close"] <= 0 or b["close"] <= 0:
                return False
            r1 = (a["close"] - b["close"]) / a["close"]
            r2 = (b["close"] - c["close"]) / b["close"]
            return r1 >= self.min_close_decrease_ratio and r2 >= self.min_close_decrease_ratio
        return False


# ========================
# 持续形态
# ========================
class RisingThreePattern(CandlePatternStrategy):
    default_params = {
        "max_inner_body_ratio": 0.8,  # 中间3根K线实体不大于首根K线的80%
        "min_final_close_above_first": 0.0,  # 最后一根至少收高于第一根的多少（比例）
    }

    param_space = {
        "max_inner_body_ratio": [0.5, 0.6, 0.8, 1.0],
        "min_final_close_above_first": [0.0, 0.01, 0.02]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return 5

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < 5:
            return None

        k = context.recent_klines[-5:]
        first, mid, last = k[0], k[1:4], k[4]

        if not (is_bullish(first) and is_bullish(last)):
            return None

        # 中间K线实体范围应小于第一根
        for m in mid:
            if candle_range(m) >= candle_range(first) * self.params["max_inner_body_ratio"]:
                return None

        # 中间3根的收盘在第一根收盘下方、开盘在最后一根开盘上方
        if not all(m["close"] < first["close"] and m["open"] > last["open"] for m in mid):
            return None

        # 最后一根应有效上破第一根
        min_diff = self.params["min_final_close_above_first"]
        if last["close"] > first["close"] * (1 + min_diff):
            strength = (last["close"] - first["close"]) / (first["close"] + 1e-9)
            strength = max(0.0, min(strength, 1.0))
            return self.build_signal(
                kline=last,
                signal_type=SignalType.BUY,
                strength=strength,
                metadata={"pattern": "RisingThree"},
                expected_action="open_long"
            )
        return None


class FallingThreePattern(CandlePatternStrategy):
    default_params = {
        "max_inner_body_ratio": 0.8,  # 中间3根K线实体不大于首根K线的80%
        "min_final_close_below_first": 0.0,  # 最后一根至少收低于第一根的多少（比例）
    }

    param_space = {
        "max_inner_body_ratio": [0.5, 0.6, 0.8, 1.0],
        "min_final_close_below_first": [0.0, 0.01, 0.02]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return 5

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < 5:
            return None

        k = context.recent_klines[-5:]
        first, mid, last = k[0], k[1:4], k[4]

        if not (is_bearish(first) and is_bearish(last)):
            return None

        for m in mid:
            if candle_range(m) >= candle_range(first) * self.params["max_inner_body_ratio"]:
                return None

        if not all(m["close"] > first["close"] and m["open"] < last["open"] for m in mid):
            return None

        min_diff = self.params["min_final_close_below_first"]
        if last["close"] < first["close"] * (1 - min_diff):
            strength = (first["close"] - last["close"]) / (first["close"] + 1e-9)
            strength = max(0.0, min(strength, 1.0))
            return self.build_signal(
                kline=last,
                signal_type=SignalType.SELL,
                strength=strength,
                metadata={"pattern": "FallingThree"},
                expected_action="open_short"
            )
        return None


# ========================
# 十字星
# ========================

class DojiPattern(CandlePatternStrategy):
    default_params = {
        "doji_threshold": 0.1,
        "signal_strength": 0.5,
        "trend_filter_enabled": True,
    }

    param_space = {
        "doji_threshold": [0.05, 0.1, 0.2],
        "signal_strength": [0.3, 0.5, 0.7, 1.0],
        "trend_filter_enabled": [True, False],
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[Signal]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None

        if is_doji(k, self.params["doji_threshold"]):
            if self.debug:
                print(f"[{k['timestamp']}] 检测 DojiPattern")
            return self.build_signal(
                kline=k,
                signal_type=SignalType.HOLD,
                strength=self.params["signal_strength"],
                metadata={"reason": "DojiPattern detected"}
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < 1:
            return None
        
        if self.params["trend_filter_enabled"]:
            if context and context.is_uptrend(): 
                return None

        return self.is_pattern(context.recent_klines[-1:])


class DragonflyDojiPattern(CandlePatternStrategy):
    default_params = {
        "min_lower_shadow_ratio": 0.6,
        "doji_threshold": 0.1,
        "signal_strength_base": 1.0,
        "signal_strength_cap": 5.0,
        "trend_filter_enabled": True,
        "trend_window": 3,
    }

    param_space = {
        "min_lower_shadow_ratio": [0.5, 0.6, 0.7],
        "doji_threshold": [0.05, 0.1],
        "signal_strength_base": [1.0, 2.0, 3.0],
        "signal_strength_cap": [3.0, 5.0, 7.0],
        "trend_filter_enabled": [True, False],
        "trend_window": [3, 5],
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return self.params["trend_window"]

    def is_pattern(self, klines: List[dict]) -> Optional[Signal]:
        k = klines[-1]
        r = candle_range(k)
        ls = lower_shadow(k)
        if r == 0:
            return None

        if not is_doji(k, self.params["doji_threshold"]):
            return None
        if ls / r < self.params["min_lower_shadow_ratio"]:
            return None
        if self.params["trend_filter_enabled"] and not is_downtrend(klines, self.params["trend_window"]):
            return None

        strength = min(ls / r * self.params["signal_strength_base"], self.params["signal_strength_cap"])
        if self.debug:
            print(f"[{k['timestamp']}] 检测 DragonflyDojiPattern")
            print(f"  lower_shadow = {ls:.4f}, range = {r:.4f}, strength = {strength:.2f}")
        return self.build_signal(
            kline=k,
            signal_type=SignalType.BUY,
            strength=strength,
            metadata={
                "reason": "DragonflyDojiPattern detected",
                "lower_shadow": ls,
                "range": r
            }
        )

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-self.params["trend_window"]:])


class GravestoneDojiPattern(CandlePatternStrategy):
    default_params = {
        "min_upper_shadow_ratio": 0.6,
        "doji_threshold": 0.1,
        "signal_strength_base": 1.0,
        "signal_strength_cap": 5.0,
        "trend_filter_enabled": True,
        "trend_window": 3,
    }

    param_space = {
        "min_upper_shadow_ratio": [0.5, 0.6, 0.7],
        "doji_threshold": [0.05, 0.1],
        "signal_strength_base": [1.0, 2.0, 3.0],
        "signal_strength_cap": [3.0, 5.0, 7.0],
        "trend_filter_enabled": [True, False],
        "trend_window": [3, 5],
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return self.params["trend_window"]

    def is_pattern(self, klines: List[dict]) -> Optional[Signal]:
        k = klines[-1]
        r = candle_range(k)
        us = upper_shadow(k)
        if r == 0:
            return None

        if not is_doji(k, self.params["doji_threshold"]):
            return None
        if us / r < self.params["min_upper_shadow_ratio"]:
            return None
        if self.params["trend_filter_enabled"] and not is_uptrend(klines, self.params["trend_window"]):
            return None

        strength = min(us / r * self.params["signal_strength_base"], self.params["signal_strength_cap"])
        if self.debug:
            print(f"[{k['timestamp']}] 检测 GravestoneDojiPattern")
            print(f"  upper_shadow = {us:.4f}, range = {r:.4f}, strength = {strength:.2f}")
        return self.build_signal(
            kline=k,
            signal_type=SignalType.SELL,
            strength=strength,
            metadata={
                "reason": "GravestoneDojiPattern detected",
                "upper_shadow": us,
                "range": r
            }
        )

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-self.params["trend_window"]:])


# ========================
# 额外形态
# ========================
class BullishHaramiPattern(CandlePatternStrategy):
    default_params = {
        "max_body_ratio": 0.5,
        "signal_strength": 1.0,
        "min_trend_strength": 0.5
    }

    param_space = {
        "max_body_ratio": [0.4, 0.5, 0.6],
        "signal_strength": [0.5, 1.0, 1.5],
        "min_trend_strength": [0.4, 0.5, 0.6]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict], context: Optional[MarketContext]) -> Optional[Signal]:
        prev, curr = klines[-2], klines[-1]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])

        if context and context.trend_strength() < self.params["min_trend_strength"]:
            return None

        if is_bearish(prev) and is_bullish(curr):
            if curr_body <= self.params["max_body_ratio"] * prev_body:
                if curr["open"] > prev["close"] and curr["close"] < prev["open"]:
                    if self.debug:
                        print(f"[{curr['timestamp']}] 检测 BullishHaramiPattern")
                    return self.build_signal(
                        kline=curr,
                        signal_type=SignalType.BUY,
                        strength=self.params["signal_strength"],
                        metadata={
                            "reason": "BullishHaramiPattern detected",
                            "prev_body": prev_body,
                            "curr_body": curr_body
                        }
                    )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-2:], context)


class BearishHaramiPattern(CandlePatternStrategy):
    default_params = {
        "max_body_ratio": 0.5,
        "signal_strength": 1.0,
        "min_trend_strength": 0.5  # 新增的趋势强度阈值参数
    }

    param_space = {
        "max_body_ratio": [0.4, 0.5, 0.6],
        "signal_strength": [0.5, 1.0, 1.5],
        "min_trend_strength": [0.4, 0.5, 0.6]  # 新增的趋势强度参数选择
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict], context: Optional[MarketContext]) -> Optional[Signal]:
        prev, curr = klines[-2], klines[-1]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])
        
        # 新增趋势强度过滤
        if context and context.trend_strength < self.params["min_trend_strength"]:
            return None
        
        if is_bullish(prev) and is_bearish(curr):
            if curr_body <= self.params["max_body_ratio"] * prev_body:
                if curr["open"] < prev["close"] and curr["close"] > prev["open"]:
                    if self.debug:
                        print(f"[{curr['timestamp']}] 检测 BearishHaramiPattern")
                    return self.build_signal(
                        kline=curr,
                        signal_type=SignalType.SELL,
                        strength=self.params["signal_strength"],
                        metadata={
                            "reason": "BearishHaramiPattern detected",
                            "prev_body": prev_body,
                            "curr_body": curr_body
                        }
                    )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-2:], context)


class TweezerBottomPattern(CandlePatternStrategy):
    default_params = {
        "max_low_diff": 0.001,
        "signal_strength": 1.0,
        "min_trend_strength": 0.5  # 新增的趋势强度阈值参数
    }

    param_space = {
        "max_low_diff": [0.001, 0.002, 0.003],
        "signal_strength": [0.5, 1.0, 1.5],
        "min_trend_strength": [0.4, 0.5, 0.6]  # 新增的趋势强度参数选择
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict], context: Optional[MarketContext]) -> Optional[Signal]:
        prev, curr = klines[-2], klines[-1]
        low_diff = abs(prev["low"] - curr["low"]) / max(prev["low"], curr["low"])
        
        # 新增趋势强度过滤
        if context and context.trend_strength < self.params["min_trend_strength"]:
            return None
        
        if low_diff <= self.params["max_low_diff"] and is_bearish(prev) and is_bullish(curr):
            if self.debug:
                print(f"[{curr['timestamp']}] 检测 TweezerBottomPattern")
            return self.build_signal(
                kline=curr,
                signal_type=SignalType.BUY,
                strength=self.params["signal_strength"],
                metadata={
                    "reason": "TweezerBottomPattern detected",
                    "low_diff": low_diff
                }
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-2:], context)


class TweezerTopPattern(CandlePatternStrategy):
    default_params = {
        "max_high_diff": 0.001,
        "signal_strength": 1.0,
        "min_trend_strength": 0.5  # 新增的趋势强度阈值参数
    }

    param_space = {
        "max_high_diff": [0.001, 0.002, 0.003],
        "signal_strength": [0.5, 1.0, 1.5],
        "min_trend_strength": [0.4, 0.5, 0.6]  # 新增的趋势强度参数选择
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict], context: Optional[MarketContext]) -> Optional[Signal]:
        prev, curr = klines[-2], klines[-1]
        high_diff = abs(prev["high"] - curr["high"]) / max(prev["high"], curr["high"])
        
        # 新增趋势强度过滤
        if context and context.trend_strength < self.params["min_trend_strength"]:
            return None
        
        if high_diff <= self.params["max_high_diff"] and is_bullish(prev) and is_bearish(curr):
            if self.debug:
                print(f"[{curr['timestamp']}] 检测 TweezerTopPattern")
            return self.build_signal(
                kline=curr,
                signal_type=SignalType.SELL,
                strength=self.params["signal_strength"],
                metadata={
                    "reason": "TweezerTopPattern detected",
                    "high_diff": high_diff
                }
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-2:], context)


# ========================
# 不确定形态
# ========================

class InsideBarPattern(CandlePatternStrategy):
    signal_type = SignalType.HOLD
    default_params = {}
    param_space = {}

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: list) -> Optional[Signal]:
        prev, curr = klines[-2], klines[-1]
        if curr["high"] <= prev["high"] and curr["low"] >= prev["low"]:
            if self.debug:
                print(f"[{curr['timestamp']}] 检测 InsideBarPattern")
                print(f"  prev high/low = {prev['high']}/{prev['low']}, curr high/low = {curr['high']}/{curr['low']}")
            return self.build_signal(
                kline=curr,
                signal_type=SignalType.HOLD,
                strength=0.3,
                metadata={"reason": "Inside bar", "prev": prev, "curr": curr}
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-2:])


class MarubozuPattern(CandlePatternStrategy):
    default_params = {"min_body_ratio": 0.95}
    param_space = {"min_body_ratio": [0.9, 0.95, 0.98, 0.99]}

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: list) -> Optional[Signal]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None

        body_len = abs(k["close"] - k["open"])
        upper_shadow_len = k["high"] - max(k["close"], k["open"])
        lower_shadow_len = min(k["close"], k["open"]) - k["low"]

        body_ratio = body_len / r
        if body_ratio >= self.params.get("min_body_ratio", 0.95):
            signal_type = SignalType.BUY if is_bullish(k) else SignalType.SELL
            if self.debug:
                print(f"[{k['timestamp']}] 检测 MarubozuPattern")
                print(f"  body_ratio = {body_ratio:.4f}")
            return self.build_signal(
                kline=k,
                signal_type=signal_type,
                strength=1.0,
                metadata={
                    "reason": "Marubozu pattern",
                    "body_ratio": body_ratio,
                    "upper_shadow": upper_shadow_len,
                    "lower_shadow": lower_shadow_len
                }
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-1:])

class SpinningTopPattern(CandlePatternStrategy):
    signal_type = SignalType.HOLD
    default_params = {"max_body_ratio": 0.3, "min_shadow_ratio": 0.3}
    param_space = {
        "max_body_ratio": [0.2, 0.25, 0.3, 0.35],
        "min_shadow_ratio": [0.25, 0.3, 0.35, 0.4]
    }

    def __init__(self, symbol: str, max_body_ratio=0.3, min_shadow_ratio=0.3, **kwargs):
        super().__init__(symbol, max_body_ratio=max_body_ratio, min_shadow_ratio=min_shadow_ratio, **kwargs)
        self.max_body_ratio = self.params.get("max_body_ratio", max_body_ratio)
        self.min_shadow_ratio = self.params.get("min_shadow_ratio", min_shadow_ratio)

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: list) -> Optional[Signal]:
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

        if body_ratio <= self.max_body_ratio and \
           upper_shadow_ratio >= self.min_shadow_ratio and \
           lower_shadow_ratio >= self.min_shadow_ratio:
            if self.debug:
                print(f"[{k['timestamp']}] 检测 SpinningTopPattern")
                print(f"  body_ratio = {body_ratio:.4f}, upper_shadow = {upper_shadow_ratio:.4f}, lower_shadow = {lower_shadow_ratio:.4f}")
            return self.build_signal(
                kline=k,
                signal_type=SignalType.HOLD,
                strength=0.3,
                metadata={
                    "reason": "Spinning top pattern",
                    "body_ratio": body_ratio
                }
            )
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-1:])

class ThreeLineStrikePattern(CandlePatternStrategy):
    default_params = {}
    param_space = {}

    def required_candles(self) -> int:
        return 4

    def is_pattern(self, klines: list) -> Optional[Signal]:
        k1, k2, k3, k4 = klines[-4], klines[-3], klines[-2], klines[-1]

        bullish_three = all(is_bullish(k) for k in [k1, k2, k3])
        bearish_three = all(is_bearish(k) for k in [k1, k2, k3])
        if not (bullish_three or bearish_three):
            return None

        def body_range(k):
            return min(k["open"], k["close"]), max(k["open"], k["close"])

        k4_body_min, k4_body_max = body_range(k4)
        k123_open_close = [k["open"] for k in [k1, k2, k3]] + [k["close"] for k in [k1, k2, k3]]
        k123_min = min(k123_open_close)
        k123_max = max(k123_open_close)

        if bullish_three and is_bearish(k4) and k4_body_min <= k123_min and k4_body_max >= k123_max:
            signal_type = SignalType.SELL
        elif bearish_three and is_bullish(k4) and k4_body_min <= k123_min and k4_body_max >= k123_max:
            signal_type = SignalType.BUY
        else:
            return None

        if self.debug:
            print(f"[{k4['timestamp']}] 检测 ThreeLineStrikePattern")
            print(f"  k4 body = ({k4_body_min}, {k4_body_max}), k123 body范围 = ({k123_min}, {k123_max})")

        return self.build_signal(
            kline=k4,
            signal_type=signal_type,
            strength=1.0,
            metadata={
                "reason": "Three Line Strike pattern",
                "body_min": k4_body_min,
                "body_max": k4_body_max
            }
        )

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if context is None or len(context.recent_klines) < self.required_candles():
            return None
        return self.is_pattern(context.recent_klines[-4:])
