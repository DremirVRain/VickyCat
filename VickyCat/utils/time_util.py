from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

MOCK_NOW = None

def now():
    return MOCK_NOW or datetime.now()

def convert_to_eastern(dt):
    beijing_tz = ZoneInfo("Asia/Shanghai")
    eastern_tz = ZoneInfo("US/Eastern")

    # 如果是字符串，先解析为 datetime（假设是北京时间字符串）
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=beijing_tz)
    elif isinstance(dt, datetime) and dt.tzinfo is None:
        # 如果是无时区 datetime，假设为北京时间，附加时区
        dt = dt.replace(tzinfo=beijing_tz)

    # 转换为美东时间
    eastern_dt = dt.astimezone(eastern_tz)
    return eastern_dt.strftime("%Y-%m-%d %H:%M:%S")
