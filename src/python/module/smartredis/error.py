# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
