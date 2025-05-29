from strategy.base_strategy import BaseStrategy
from strategy.strategy_signal import Signal, SignalType
from typing import Optional, List, Dict
from datetime import datetime

# ========================
# 工具函数（建议移入 utils.py）
# ========================

def is_bullish(k):
    """阳线判断"""
    return k["close"] > k["open"]

def is_bearish(k):
    """阴线判断"""
    return k["close"] < k["open"]

def body(k):
    """实体长度"""
    return abs(k["close"] - k["open"])

def candle_range(k):
    """整根K线长度（高低价差）"""
    return k["high"] - k["low"]

def upper_shadow(k):
    """上影线长度"""
    return k["high"] - max(k["close"], k["open"])

def lower_shadow(k):
    """下影线长度"""
    return min(k["close"], k["open"]) - k["low"]

def body_ratio(k):
    """实体占整根K线比例"""
    r = candle_range(k)
    return body(k) / r if r > 0 else 0

def upper_shadow_ratio(k):
    """上影线占整根K线比例"""
    r = candle_range(k)
    return upper_shadow(k) / r if r > 0 else 0

def lower_shadow_ratio(k):
    """下影线占整根K线比例"""
    r = candle_range(k)
    return lower_shadow(k) / r if r > 0 else 0

def is_doji(k, max_body_ratio=0.1):
    """十字星判断，默认实体不超过10%"""
    return body_ratio(k) <= max_body_ratio

def is_marubozu(k, min_body_ratio=0.9, max_upper_shadow_ratio=0.05, max_lower_shadow_ratio=0.05):
    """光头光脚线判断"""
    return (
        body_ratio(k) >= min_body_ratio and
        upper_shadow_ratio(k) <= max_upper_shadow_ratio and
        lower_shadow_ratio(k) <= max_lower_shadow_ratio
    )

def is_inside_bar(current_k, prev_k):
    """内包线判断，当前K线被上一根完全包裹"""
    return (
        current_k["high"] <= prev_k["high"] and
        current_k["low"] >= prev_k["low"]
    )

def gap_up(current_k, prev_k):
    """跳空高开"""
    return current_k["low"] > prev_k["high"]

def gap_down(current_k, prev_k):
    """跳空低开"""
    return current_k["high"] < prev_k["low"]

def is_engulfing(current_k, prev_k, bullish=True):
    """
    吞没形态判断
    bullish=True判断看涨吞没（阳线包阴线）
    bullish=False判断看跌吞没（阴线包阳线）
    """
    if bullish:
        return (
            is_bullish(current_k) and
            is_bearish(prev_k) and
            current_k["open"] < prev_k["close"] and
            current_k["close"] > prev_k["open"]
        )
    else:
        return (
            is_bearish(current_k) and
            is_bullish(prev_k) and
            current_k["open"] > prev_k["close"] and
            current_k["close"] < prev_k["open"]
        )
    

class CandlePatternStrategy(BaseStrategy):
    signal_type: Optional[SignalType] = None

    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.cache = {}  # 如果子类有需要，也可利用

    def required_candles(self) -> int:
        raise NotImplementedError

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        raise NotImplementedError

    def generate_signal(self, *klines: dict) -> Optional[Signal]:
        required = self.required_candles()
        if len(klines) < required:
            return None
        selected_klines = klines[-required:]
        signal_type = self.is_pattern(selected_klines)
        if signal_type:
            return self.build_signal(
                kline=selected_klines[-1],
                signal_type=signal_type,
                metadata={"pattern": self.__class__.__name__}
            )
        return None

    def get_params(self) -> dict:
        return {}

    def set_params(self, **params):
        pass

# ========================
# 单K线反转形态策略
# ========================

class HammerPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, min_body_ratio=0.2, min_shadow_ratio=2.0, max_upper_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.max_upper_shadow_ratio = max_upper_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        ls = lower_shadow(k)
        us = upper_shadow(k)

        if r == 0:
            return None

        if (
            b / r >= self.min_body_ratio and
            ls > self.min_shadow_ratio * b and
            us < self.max_upper_shadow_ratio * r and
            is_bullish(k)
        ):
            print(f"✅ 锤子线匹配: {k}")
            print(f"[{k['timestamp']}] 检测 HammerPattern")
            print(f"  body = {b:.4f}, range = {r:.4f}, body/range = {b/r:.4f}")
            print(f"  lower_shadow = {ls:.4f} vs {self.min_shadow_ratio} * body = {self.min_shadow_ratio * b:.4f}")
            print(f"  upper_shadow = {us:.4f} vs {self.max_upper_shadow_ratio} * range = {self.max_upper_shadow_ratio * r:.4f}")
            print(f"  is_bullish = {is_bullish(k)}")
            return self.signal_type
        return None



class HangingManPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, min_body_ratio=0.2, min_shadow_ratio=2.0, max_upper_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.max_upper_shadow_ratio = max_upper_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        ls = lower_shadow(k)
        us = upper_shadow(k)

        if r == 0:
            return None

        if (
            b / r >= self.min_body_ratio and
            ls > self.min_shadow_ratio * b and
            us < self.max_upper_shadow_ratio * r and
            is_bearish(k)
        ):
            print(f"✅ 吊人线匹配: {k}")
            print(f"[{k['timestamp']}] 检测 HangingManPattern")
            print(f"  body = {b:.4f}, range = {r:.4f}, body/range = {b/r:.4f}")
            print(f"  lower_shadow = {ls:.4f} vs {self.min_shadow_ratio} * body = {self.min_shadow_ratio * b:.4f}")
            print(f"  upper_shadow = {us:.4f} vs {self.max_upper_shadow_ratio} * range = {self.max_upper_shadow_ratio * r:.4f}")
            print(f"  is_bearish = {is_bearish(k)}")
            return self.signal_type
        return None



class InvertedHammerPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, min_body_ratio=0.2, min_shadow_ratio=2.0, max_lower_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.max_lower_shadow_ratio = max_lower_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        ls = lower_shadow(k)
        us = upper_shadow(k)

        if r == 0:
            return None

        if (
            b / r >= self.min_body_ratio and
            us > self.min_shadow_ratio * b and
            ls < self.max_lower_shadow_ratio * r and
            is_bullish(k)
        ):
            print(f"✅ 倒锤子线匹配: {k}")
            print(f"[{k['timestamp']}] 检测 InvertedHammerPattern")
            print(f"  body = {b:.4f}, range = {r:.4f}, body/range = {b/r:.4f}")
            print(f"  upper_shadow = {us:.4f} vs {self.min_shadow_ratio} * body = {self.min_shadow_ratio * b:.4f}")
            print(f"  lower_shadow = {ls:.4f} vs {self.max_lower_shadow_ratio} * range = {self.max_lower_shadow_ratio * r:.4f}")
            print(f"  is_bullish = {is_bullish(k)}")
            return self.signal_type
        return None



class ShootingStarPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, min_body_ratio=0.2, min_shadow_ratio=2.0, max_lower_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.max_lower_shadow_ratio = max_lower_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        ls = lower_shadow(k)
        us = upper_shadow(k)

        if r == 0:
            return None

        if (
            b / r >= self.min_body_ratio and
            us > self.min_shadow_ratio * b and
            ls < self.max_lower_shadow_ratio * r and
            is_bearish(k)
        ):
            print(f"✅ 流星线匹配: {k}")
            print(f"[{k['timestamp']}] 检测 ShootingStarPattern")
            print(f"  body = {b:.4f}, range = {r:.4f}, body/range = {b/r:.4f}")
            print(f"  upper_shadow = {us:.4f} vs {self.min_shadow_ratio} * body = {self.min_shadow_ratio * b:.4f}")
            print(f"  lower_shadow = {ls:.4f} vs {self.max_lower_shadow_ratio} * range = {self.max_lower_shadow_ratio * r:.4f}")
            print(f"  is_bearish = {is_bearish(k)}")
            return self.signal_type
        return None


# ========================
# 双K线反转形态
# ========================

class BullishEngulfingPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, open_close_overlap_ratio=0.0):
        super().__init__(symbol)
        self.open_close_overlap_ratio = open_close_overlap_ratio  # 允许开盘价包容的比例（0为完全包容）

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        if is_bearish(prev) and is_bullish(curr):
            # curr开盘价必须低于prev收盘价 * (1 + overlap_ratio)，close必须大于prev开盘价 * (1 - overlap_ratio)
            if curr["open"] < prev["close"] * (1 + self.open_close_overlap_ratio) and \
               curr["close"] > prev["open"] * (1 - self.open_close_overlap_ratio):
                return self.signal_type
        return None


class BearishEngulfingPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, open_close_overlap_ratio=0.0):
        super().__init__(symbol)
        self.open_close_overlap_ratio = open_close_overlap_ratio

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        if is_bullish(prev) and is_bearish(curr):
            if curr["open"] > prev["close"] * (1 - self.open_close_overlap_ratio) and \
               curr["close"] < prev["open"] * (1 + self.open_close_overlap_ratio):
                return self.signal_type
        return None


class PiercingLinePattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, min_close_above_midpoint_ratio=0.5):
        super().__init__(symbol)
        self.min_close_above_midpoint_ratio = min_close_above_midpoint_ratio  # 最低超过前一K实体中点比例

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        midpoint = (prev["open"] + prev["close"]) / 2
        if is_bearish(prev) and is_bullish(curr):
            if curr["open"] < prev["low"] and \
               curr["close"] > midpoint + self.min_close_above_midpoint_ratio * abs(prev["close"] - prev["open"]):
                return self.signal_type
        return None


class DarkCloudCoverPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, max_close_below_midpoint_ratio=0.5):
        super().__init__(symbol)
        self.max_close_below_midpoint_ratio = max_close_below_midpoint_ratio

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        midpoint = (prev["open"] + prev["close"]) / 2
        if is_bullish(prev) and is_bearish(curr):
            if curr["open"] > prev["high"] and \
               curr["close"] < midpoint - self.max_close_below_midpoint_ratio * abs(prev["close"] - prev["open"]):
                return self.signal_type
        return None

# ========================
# 三K线反转形态
# ========================

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

class DojiPattern(CandlePatternStrategy):
    signal_type = None  # 中性信号

    def __init__(self, symbol: str, max_body_ratio=0.1):
        super().__init__(symbol)
        self.max_body_ratio = max_body_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None
        b = body(k)
        if b / r <= self.max_body_ratio:
            return self.signal_type
        return None


class DragonflyDojiPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, max_body_ratio=0.1, lower_shadow_ratio=0.6):
        super().__init__(symbol)
        self.max_body_ratio = max_body_ratio
        self.lower_shadow_ratio = lower_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None
        b = body(k)
        ls = lower_shadow(k)
        if b / r <= self.max_body_ratio and ls / r >= self.lower_shadow_ratio:
            return self.signal_type
        return None


class GravestoneDojiPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, max_body_ratio=0.1, upper_shadow_ratio=0.6):
        super().__init__(symbol)
        self.max_body_ratio = max_body_ratio
        self.upper_shadow_ratio = upper_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        r = candle_range(k)
        if r == 0:
            return None
        b = body(k)
        us = upper_shadow(k)
        if b / r <= self.max_body_ratio and us / r >= self.upper_shadow_ratio:
            return self.signal_type
        return None

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
