import traceback

class ErrorHandler:
    @staticmethod
    def log_error(message: str, exception: Exception):
        """ 统一异常处理方法 """
        error_msg = f"[Error] {message}: {exception}"
        print(error_msg)
        print(traceback.format_exc())