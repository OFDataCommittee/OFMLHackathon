class RuntimeError(Exception):
    """Base client runtime error"""

    def __init__(self, message):
        self.msg = message

    def __str__(self):
        return self.msg


class ConnectionError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
