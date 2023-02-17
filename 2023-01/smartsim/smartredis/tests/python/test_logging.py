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

from smartredis import *
from smartredis.error import *
import pytest

@pytest.mark.parametrize("log_level", [
    LLQuiet, LLInfo, LLDebug, LLDeveloper
])
def test_logging_string(use_cluster, context, log_level):
    log_data(context, log_level, f"This is data logged from a string ({log_level.name})")
    log_warning(context, log_level, f"This is a warning logged from a string ({log_level.name})")
    log_error(context, log_level, f"This is an error logged from a string ({log_level.name})")

@pytest.mark.parametrize("log_level", [
    LLQuiet, LLInfo, LLDebug, LLDeveloper
])
def test_logging_client(use_cluster, context, log_level):
    c = Client(None, use_cluster, logger_name=context)
    c.log_data(log_level, f"This is data logged from a client ({log_level.name})")
    c.log_warning(log_level, f"This is a warning logged from a client ({log_level.name})")
    c.log_error(log_level, f"This is an error logged from a client ({log_level.name})")

@pytest.mark.parametrize("log_level", [
    LLQuiet, LLInfo, LLDebug, LLDeveloper
])
def test_logging_dataset(context, log_level):
    d = Dataset(context)
    d.log_data(log_level, f"This is data logged from a dataset ({log_level.name})")
    d.log_warning(log_level, f"This is a warning logged from a dataset ({log_level.name})")
    d.log_error(log_level, f"This is an error logged from a dataset ({log_level.name})")

@pytest.mark.parametrize("log_level", [
    LLQuiet, LLInfo, LLDebug, LLDeveloper
])
def test_logging_logcontext(context, log_level):
    lc = LogContext(context)
    lc.log_data(log_level, f"This is data logged from a logcontext ({log_level.name})")
    lc.log_warning(log_level, f"This is a warning logged from a logcontext ({log_level.name})")
    lc.log_error(log_level, f"This is an error logged from a logcontext ({log_level.name})")
