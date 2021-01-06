from os import environ


class RedisConnectionError(RuntimeError):
    def __init__(self, cpp_error=""):
        super().__init__(self._set_message(cpp_error))

    @staticmethod
    def _set_message(cpp_error):
        msg = ""
        if cpp_error:
            msg = cpp_error + "\n"
        if "SSDB" in environ:
            msg += f"Could not connect to Redis at {environ['SSDB']}"
            return msg
        msg += "Could not connect to database. $SSDB not set"
        return msg


class RedisReplyError(RuntimeError):
    def __init__(self, cpp_error, method, key=""):
        super().__init__(self._check_error(cpp_error, method, key))

    @staticmethod
    def _check_error(cpp_error, method, key):
        msg = f"Client.{method} execution failed\n"
        if "REDIS_REPLY_NIL" in cpp_error:
            msg += f"No Dataset stored at key: {key}"
            return msg
        msg += cpp_error
        return msg
