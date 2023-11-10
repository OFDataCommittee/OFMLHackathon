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

import os
import pytest

from smartredis import Client, ConfigOptions

def test_serialization(context):
    c = Client(None, logger_name=context)
    assert str(c) != repr(c)


def test_address(context):
    # get env var to set through client init
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]

    # client init should fail if SSDB not set
    _ = Client(False, address=ssdb, logger_name=context)

    # check if SSDB was set anyway
    assert os.environ["SSDB"] == ssdb

# Globals for Client constructor testing
ac_original = Client._Client__address_construction
sc_original = Client._Client__standard_construction
cluster_mode = os.environ["SR_DB_TYPE"] == "Clustered"
target_address = os.environ["SSDB"]
co_envt = ConfigOptions.create_from_environment("")

@pytest.mark.parametrize(
    "args, kwargs, expected_constructor", [
    # address constructions
    [(False,), {}, "address"],
    [(False,), {"address": target_address}, "address"],
    [(False,), {"address": target_address, "logger_name": "log_name"}, "address"],
    [(False,), {"logger_name": "log_name"}, "address"],
    [(False, target_address), {}, "address"],
    [(False, target_address), {"logger_name": "log_name"}, "address"],
    [(False, target_address, "log_name"), {}, "address"],
    [(), {"cluster": cluster_mode}, "address"],
    [(), {"cluster": cluster_mode, "address": target_address}, "address"],
    [(), {"cluster": cluster_mode, "address": target_address, "logger_name": "log_name"}, "address"],
    [(), {"cluster": cluster_mode, "logger_name": "log_name"}, "address"],
    # standard constructions
    [(None,), {}, "standard"],
    [(None,), {"logger_name": "log_name"}, "standard"],
    [(None, "log_name"), {}, "standard"],
    [(co_envt,), {}, "standard"],
    [(co_envt,), {"logger_name": "log_name"}, "standard"],
    [(co_envt, "log_name"), {}, "standard"],
    [(), {}, "standard"],
    [(), {"config_options": None}, "standard"],
    [(), {"config_options": None, "logger_name": "log_name"}, "standard"],
    [(), {"config_options": co_envt}, "standard"],
    [(), {"config_options": co_envt, "logger_name": "log_name"}, "standard"],
    [(), {"logger_name": "log_name"}, "standard"],
])
def test_client_constructor(args, kwargs, expected_constructor, monkeypatch):
    ac_got_called = False
    sc_got_called = False

    def mock_address_constructor(self, *a, **kw):
        nonlocal ac_got_called
        ac_got_called = True
        return ac_original(self, *a, **kw)

    @staticmethod
    def mock_standard_constructor(*a, **kw):
        nonlocal sc_got_called
        sc_got_called = True
        return sc_original(*a, **kw)

    monkeypatch.setattr(
        Client, "_Client__address_construction", mock_address_constructor)
    monkeypatch.setattr(
        Client, "_Client__standard_construction", mock_standard_constructor)

    Client(*args, **kwargs)

    if expected_constructor == "address":
        assert ac_got_called
        assert not sc_got_called
    if expected_constructor == "standard":
        assert not ac_got_called
        assert sc_got_called
