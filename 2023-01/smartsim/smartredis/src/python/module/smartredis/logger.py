# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

from .smartredisPy import SRLoggingLevel, cpp_log_data, cpp_log_error, cpp_log_warning
from .util import exception_handler, typecheck

# Logging levels
# LLQuiet     = 1  # No logging at all
# LLInfo      = 2  # Informational logging only
# LLDebug     = 3  # Verbose logging for debugging purposes
# LLDeveloper = 4  # Extra verbose logging for internal use


@exception_handler
def log_data(context: str, level: SRLoggingLevel, data: str) -> None:
    """Log data to the SmartRedis logfile

    :param context: Logging context (string to prefix the log entry with)
    :type context: str
    :param level: minimum logging level for data to be logged with
    :type name: enum
    :param data: message data to log
    :type data: str
    :raises RedisReplyError: if logging fails
    """
    typecheck(context, "context", str)
    typecheck(level, "level", SRLoggingLevel)
    typecheck(data, "data", str)
    cpp_log_data(context, level, data)


@exception_handler
def log_warning(context: str, level: SRLoggingLevel, data: str) -> None:
    """Log a warning to the SmartRedis logfile

    :param context: Logging context (string to prefix the log entry with)
    :type context: str
    :param level: minimum logging level for data to be logged with
    :type name: enum
    :param data: message data to log
    :type data: str
    :raises RedisReplyError: if logging fails
    """
    typecheck(context, "context", str)
    typecheck(level, "level", SRLoggingLevel)
    typecheck(data, "data", str)
    cpp_log_warning(context, level, data)


@exception_handler
def log_error(context: str, level: SRLoggingLevel, data: str) -> None:
    """Log an error to the SmartRedis logfile

    :param context: Logging context (string to prefix the log entry with)
    :type context: str
    :param level: minimum logging level for data to be logged with
    :type name: enum
    :param data: message data to log
    :type data: str
    :raises RedisReplyError: if logging fails
    """
    typecheck(context, "context", str)
    typecheck(level, "level", SRLoggingLevel)
    typecheck(data, "data", str)
    cpp_log_error(context, level, data)
