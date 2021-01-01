from os import environ


class RedisConnectionError(RuntimeError):
    def __init__(self, cpp_error):
        self.msg = self._set_message()
        self.error_from_cpp = cpp_error

    def __str__(self):
        return "\n".join((self.error_from_cpp, self.msg))

    def _set_message(self):
        if environ["SSDB"]:
            return f"Could not connect to SSDB at {environ['SSDB']}"
        return "Could not connect to database. $SSDB not set"

class RedisReplyError(RuntimeError):

    def __init__(self, cpp_error, key, method):
        self.msg = self._check_error(cpp_error, key, method)

    def __str__(self):
        return self.msg

    def _check_error(self, cpp_error, key, method):
        msg = f"Client.{method} execution failed\n"
        if "REDIS_REPLY_NIL" in cpp_error:
            msg += f"No Dataset stored at key: {key}"
            return msg
        else:
            msg += cpp_error
            return msg