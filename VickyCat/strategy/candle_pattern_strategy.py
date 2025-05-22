from base_strategy import BaseStrategy
from strategy_signal import Signal, SignalType
from typing import Optional, List
from datetime import datetime

class CandlePatternStrategy(BaseStrategy):
    signal_type: Optional[SignalType] = None

    def __init__(self, symbol: str):
        super().__init__(symbol)

    def required_candles(self) -> int:
        raise NotImplementedError

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        raise NotImplementedError

    def generate_signal(self, *klines: dict) -> Optional[Signal]:
        if len(klines) < self.required_candles():
            return None
        signal_type = self.is_pattern(klines[-self.required_candles():])
        if signal_type:
            return Signal(self.symbol, klines[-1]["timestamp"], signal_type, strategy_name=self.__class__.__name__)
        return None

# ========================
# ���ߺ������������� utils.py��
# ========================

def is_bullish(k):
    """�����ж�"""
    return k["close"] > k["open"]

def is_bearish(k):
    """�����ж�"""
    return k["close"] < k["open"]

def body(k):
    """ʵ�峤��"""
    return abs(k["close"] - k["open"])

def candle_range(k):
    """����K�߳��ȣ��ߵͼ۲"""
    return k["high"] - k["low"]

def upper_shadow(k):
    """��Ӱ�߳���"""
    return k["high"] - max(k["close"], k["open"])

def lower_shadow(k):
    """��Ӱ�߳���"""
    return min(k["close"], k["open"]) - k["low"]

def body_ratio(k):
    """ʵ��ռ����K�߱���"""
    r = candle_range(k)
    return body(k) / r if r > 0 else 0

def upper_shadow_ratio(k):
    """��Ӱ��ռ����K�߱���"""
    r = candle_range(k)
    return upper_shadow(k) / r if r > 0 else 0

def lower_shadow_ratio(k):
    """��Ӱ��ռ����K�߱���"""
    r = candle_range(k)
    return lower_shadow(k) / r if r > 0 else 0

def is_doji(k, max_body_ratio=0.1):
    """ʮ�����жϣ�Ĭ��ʵ�岻����10%"""
    return body_ratio(k) <= max_body_ratio

def is_marubozu(k, min_body_ratio=0.9, max_upper_shadow_ratio=0.05, max_lower_shadow_ratio=0.05):
    """��ͷ������ж�"""
    return (
        body_ratio(k) >= min_body_ratio and
        upper_shadow_ratio(k) <= max_upper_shadow_ratio and
        lower_shadow_ratio(k) <= max_lower_shadow_ratio
    )

def is_inside_bar(current_k, prev_k):
    """�ڰ����жϣ���ǰK�߱���һ����ȫ����"""
    return (
        current_k["high"] <= prev_k["high"] and
        current_k["low"] >= prev_k["low"]
    )

def gap_up(current_k, prev_k):
    """���ո߿�"""
    return current_k["low"] > prev_k["high"]

def gap_down(current_k, prev_k):
    """���յͿ�"""
    return current_k["high"] < prev_k["low"]

def is_engulfing(current_k, prev_k, bullish=True):
    """
    ��û��̬�ж�
    bullish=True�жϿ�����û�����߰����ߣ�
    bullish=False�жϿ�����û�����߰����ߣ�
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

# ========================
# ��̬��
# ========================

class HammerPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, min_body_ratio=0.2, shadow_to_body_ratio=2.0, max_upper_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.shadow_to_body_ratio = shadow_to_body_ratio
        self.max_upper_shadow_ratio = max_upper_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        if r == 0:
            return None
        # ��Ӱ�߳���ʵ������������С��������Ӱ�߽϶�
        if (
            b / r >= self.min_body_ratio and
            lower_shadow(k) > self.shadow_to_body_ratio * b and
            upper_shadow(k) < self.max_upper_shadow_ratio * r and
            is_bullish(k)
        ):
            return self.signal_type
        return None


class HangingManPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, min_body_ratio=0.2, shadow_to_body_ratio=2.0, max_upper_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.shadow_to_body_ratio = shadow_to_body_ratio
        self.max_upper_shadow_ratio = max_upper_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        if r == 0:
            return None
        # ���ƴ����ߣ���ʵ��Ϊ���ߣ�����Ӱ�߳�����Ӱ�߶�
        if (
            b / r >= self.min_body_ratio and
            lower_shadow(k) > self.shadow_to_body_ratio * b and
            upper_shadow(k) < self.max_upper_shadow_ratio * r and
            is_bearish(k)
        ):
            return self.signal_type
        return None


class InvertedHammerPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, min_body_ratio=0.2, shadow_to_body_ratio=2.0, max_lower_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.shadow_to_body_ratio = shadow_to_body_ratio
        self.max_lower_shadow_ratio = max_lower_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        if r == 0:
            return None
        # ����Ӱ�ߣ�ʵ��ռ����С������Ӱ�߽϶̣�ʵ��Ϊ����
        if (
            b / r >= self.min_body_ratio and
            upper_shadow(k) > self.shadow_to_body_ratio * b and
            lower_shadow(k) < self.max_lower_shadow_ratio * r and
            is_bullish(k)
        ):
            return self.signal_type
        return None


class ShootingStarPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def __init__(self, symbol: str, min_body_ratio=0.2, shadow_to_body_ratio=2.0, max_lower_shadow_ratio=0.3):
        super().__init__(symbol)
        self.min_body_ratio = min_body_ratio
        self.shadow_to_body_ratio = shadow_to_body_ratio
        self.max_lower_shadow_ratio = max_lower_shadow_ratio

    def required_candles(self) -> int:
        return 1

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        k = klines[-1]
        b = body(k)
        r = candle_range(k)
        if r == 0:
            return None
        # ���Ƶ������ߣ���Ϊ���ߣ�����Ӱ�ߣ�����Ӱ��
        if (
            b / r >= self.min_body_ratio and
            upper_shadow(k) > self.shadow_to_body_ratio * b and
            lower_shadow(k) < self.max_lower_shadow_ratio * r and
            is_bearish(k)
        ):
            return self.signal_type
        return None

# ========================
# ˫K�߷�ת��̬
# ========================

class BullishEngulfingPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, open_close_overlap_ratio=0.0):
        super().__init__(symbol)
        self.open_close_overlap_ratio = open_close_overlap_ratio  # �����̼۰��ݵı�����0Ϊ��ȫ���ݣ�

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: List[dict]) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        if is_bearish(prev) and is_bullish(curr):
            # curr���̼۱������prev���̼� * (1 + overlap_ratio)��close�������prev���̼� * (1 - overlap_ratio)
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
        self.min_close_above_midpoint_ratio = min_close_above_midpoint_ratio  # ��ͳ���ǰһKʵ���е����

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
# ��K�߷�ת��̬
# ========================

class MorningStarPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, max_mid_body_ratio=0.3, min_third_close_ratio=0.5):
        super().__init__(symbol)
        self.max_mid_body_ratio = max_mid_body_ratio  # �м�Kʵ����������ʮ���ǻ�Сʵ�壩
        self.min_third_close_ratio = min_third_close_ratio  # ������K�����̱���ͻ��ǰKʵ��һ�����ϱ���

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
        self.min_close_increase_ratio = min_close_increase_ratio  # ���������������̼۱����������Ǳ���

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
# ������̬
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
# ʮ����
# ========================

class DojiPattern(CandlePatternStrategy):
    signal_type = None  # �����ź�

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
# ������̬
# ========================

class BullishHaramiPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def __init__(self, symbol: str, max_body_ratio=0.5):
        super().__init__(symbol)
        self.max_body_ratio = max_body_ratio  # �ڶ���K��ʵ���������ڵ�һ���ı���

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
        self.max_low_diff = max_low_diff  # ����K����ͼ����������

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
# ��ȷ����̬
# ========================

class InsideBarPattern(CandlePatternStrategy):
    signal_type = SignalType.WARNING

    def required_candles(self) -> int:
        return 2

    def is_pattern(self, klines: list) -> Optional[SignalType]:
        prev, curr = klines[-2], klines[-1]
        # ��ǰK����߼۲�����ǰK����߼ۣ���ͼ۲�����ǰK����ͼ�
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

        # ʵ��ռ������95%
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

        # �ж�ǰ�����Ƿ�ͬɫ������򶼿գ�
        bullish_three = all(is_bullish(k) for k in [k1, k2, k3])
        bearish_three = all(is_bearish(k) for k in [k1, k2, k3])
        if not (bullish_three or bearish_three):
            return None

        # ���ĸ��Ƿ����Ҵ�ʵ�帲��ǰ����������
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
