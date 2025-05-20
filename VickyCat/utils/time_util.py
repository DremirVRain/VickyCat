# utils/time_util.py
MOCK_NOW = None

def now():
    return MOCK_NOW or datetime.now()
