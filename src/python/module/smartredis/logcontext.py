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

from .smartredisPy import PyLogContext
from .srobject import SRObject
from .util import exception_handler, typecheck

from .error import *

class LogContext(SRObject):
    def __init__(self, context):
        """Initialize a LogContext object

        :param context: logging context
        :type name: str
        """
        super().__init__(PyLogContext(context))
        typecheck(context, "context", str)
        self._name = context

    @property
    def _logcontext(self):
        """Alias _srobject to _logcontext
        """
        return self._srobject

    @staticmethod
    def from_pybind(logcontext):
        """Initialize a LogContext object from
        a PyLogContext object

        :param logcontext: The pybind PyLogContext object
                           to use for construction
        :type logcontext: PyLogContext
        :return: The newly constructor LogContext from
                 the PyLogContext
        :rtype: LogContext
        """
        typecheck(logcontext, "logcontext", PyLogContext)
        new_logcontext = LogContext(logcontext._name)
        new_logcontext.set_context(logcontext)
        return new_logcontext

    @exception_handler
    def get_context(self):
        """Return the PyLogContext attribute

        :return: The PyLogContext attribute containing
                 the logcontext information
        :rtype: PyLogContext
        """
        return self._logcontext

    @exception_handler
    def set_context(self, logcontext):
        """Set the PyLogContext attribute

        :param logcontext: The PyLogContext object
        :type logcontext: PyLogContext
        """
        typecheck(logcontext, "logcontext", PyLogContext)
        self._srobject = logcontext
