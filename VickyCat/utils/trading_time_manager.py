import pytz
from datetime import datetime
from typing import List

class TradingTimeManager:
    """多市场交易时段管理器，支持夏令时自动识别"""

    # 各市场时区映射
    MARKET_TIMEZONES = {
        "US": "US/Eastern",   # 会自动处理夏令时
        "HK": "Asia/Hong_Kong",
        "CN": "Asia/Shanghai",
        "SG": "Asia/Singapore",
    }

    # 各市场交易时段（单位：HHMM，24小时制）
    MARKET_SESSIONS = {
        "US": [(930, 1600)],
        "HK": [(930, 1200), (1300, 1600)],
        "CN": [(930, 1130), (1300, 1500)],
        "SG": [(900, 1200), (1300, 1700)],
    }

    def __init__(self, market: str = "US"):
        self.market = market.upper()
        if self.market not in self.MARKET_TIMEZONES:
            raise ValueError(f"Unsupported market: {self.market}")

    def get_market_time(self) -> datetime:
        """获取当前市场时间（含夏令时判断）"""
        tz = pytz.timezone(self.MARKET_TIMEZONES[self.market])
        return datetime.now(tz)

    def is_trading_time(self) -> bool:
        """判断当前是否为交易时段"""
        now = self.get_market_time()
        current = now.hour * 100 + now.minute

        for start, end in self.MARKET_SESSIONS[self.market]:
            if start <= current <= end:
                return True
        return False

    def get_trading_periods(self) -> List[str]:
        """返回当前市场交易时段区间（字符串形式）"""
        return [f"{s:04d}-{e:04d}" for s, e in self.MARKET_SESSIONS[self.market]]
