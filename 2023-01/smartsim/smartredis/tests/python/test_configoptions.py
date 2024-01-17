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
from smartredis import *
from smartredis.error import *

#####
# Test attempts to use API functions from non-factory object

def test_non_factory_configobject():
    co = ConfigOptions()
    with pytest.raises(RedisRuntimeError):
        _ = co.get_integer_option("key")
    with pytest.raises(RedisRuntimeError):
        _ = co.get_string_option("key")
    with pytest.raises(RedisRuntimeError):
        _ = co.is_configured("key")
    with pytest.raises(RedisRuntimeError):
        _ = co.override_integer_option("key", 42)
    with pytest.raises(RedisRuntimeError):
        _ = co.override_string_option("key", "value")

def test_options(monkeypatch):
    monkeypatch.setenv("test_integer_key", "42")
    monkeypatch.setenv("test_string_key", "charizard")
    co = ConfigOptions.create_from_environment("")

    # integer option tests
    assert co.get_integer_option("test_integer_key") == 42
    assert not co.is_configured("test_integer_key_that_is_not_really_present")
    with pytest.raises(RedisKeyError):
        _ = co.get_integer_option("test_integer_key_that_is_not_really_present")
    co.override_integer_option("test_integer_key_that_is_not_really_present", 42)
    assert co.is_configured("test_integer_key_that_is_not_really_present")
    assert co.get_integer_option(
        "test_integer_key_that_is_not_really_present") == 42

    # string option tests
    assert co.get_string_option("test_string_key") == "charizard"
    assert not co.is_configured("test_string_key_that_is_not_really_present")
    with pytest.raises(RedisKeyError):
        _ = co.get_string_option("test_string_key_that_is_not_really_present")
    co.override_string_option("test_string_key_that_is_not_really_present", "meowth")
    assert co.is_configured("test_string_key_that_is_not_really_present")
    assert co.get_string_option(
        "test_string_key_that_is_not_really_present") == "meowth"

def test_options_with_suffix(monkeypatch):
    monkeypatch.setenv("integer_key_suffixtest", "42")
    monkeypatch.setenv("string_key_suffixtest", "charizard")
    co = ConfigOptions.create_from_environment("suffixtest")

    # integer option tests
    assert co.get_integer_option("integer_key") == 42
    assert not co.is_configured("integer_key_that_is_not_really_present")
    with pytest.raises(RedisKeyError):
        _ = co.get_integer_option("integer_key_that_is_not_really_present")
    co.override_integer_option("integer_key_that_is_not_really_present", 42)
    assert co.get_integer_option("integer_key_that_is_not_really_present") == 42
    assert co.is_configured("integer_key_that_is_not_really_present")

    # string option tests
    assert co.get_string_option("string_key") == "charizard"
    assert not co.is_configured("string_key_that_is_not_really_present")
    with pytest.raises(RedisKeyError):
        _ = co.get_string_option("string_key_that_is_not_really_present")
    co.override_string_option("string_key_that_is_not_really_present", "meowth")
    assert co.is_configured("string_key_that_is_not_really_present")
    assert co.get_string_option(
        "string_key_that_is_not_really_present") == "meowth"
