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
