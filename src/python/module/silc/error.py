from os import environ


class RedisConnectionError(RuntimeError):
    def __init__(self, cpp_error=""):
        self.msg = self._set_message()
        self.error_from_cpp = cpp_error

    def __str__(self):
        return "\n".join((self.error_from_cpp, self.msg))

    @staticmethod
    def _set_message():
        if "SSDB" in environ:
            return f"Could not connect to Redis at {environ['SSDB']}"
        return "Could not connect to database. $SSDB not set"


class RedisReplyError(RuntimeError):
    def __init__(self, cpp_error, method, key=""):
        self.msg = self._check_error(cpp_error, method, key)

    def __str__(self):
        return self.msg

    @staticmethod
    def _check_error(cpp_error, method, key):
        msg = f"Client.{method} execution failed\n"
        if "REDIS_REPLY_NIL" in cpp_error:
            msg += f"No Dataset stored at key: {key}"
            return msg
        msg += cpp_error
        return msg
