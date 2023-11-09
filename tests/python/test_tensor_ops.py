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
import numpy as np
import pytest
from smartredis import Client
from smartredis.error import RedisReplyError


def test_copy_tensor(context):
    # test copying tensor

    client = Client(None, logger_name=context)
    tensor = np.array([1, 2])
    client.put_tensor("test_copy", tensor)

    client.copy_tensor("test_copy", "test_copied")
    bool_poll_key = client.poll_key(get_prefix() + "test_copy", 100, 100)
    assert bool_poll_key == True
    assert client.key_exists(get_prefix() + "test_copy")
    assert client.key_exists(get_prefix() + "test_copied")
    returned = client.get_tensor("test_copied")
    assert np.array_equal(tensor, returned)


def test_rename_tensor(context):
    # test renaming tensor

    client = Client(None, logger_name=context)
    tensor = np.array([1, 2])
    client.put_tensor("test_rename", tensor)

    client.rename_tensor("test_rename", "test_renamed")

    assert not client.tensor_exists("test_rename")
    assert client.tensor_exists("test_renamed")
    returned = client.get_tensor("test_renamed")
    assert np.array_equal(tensor, returned)


def test_delete_tensor(context):
    # test renaming tensor

    client = Client(None, logger_name=context)
    tensor = np.array([1, 2])
    client.put_tensor("test_delete", tensor)

    client.delete_tensor("test_delete")

    assert not (client.key_exists("test_delete"))


# --------------- Error handling ----------------------


def test_rename_nonexisting_key(context):

    client = Client(None, logger_name=context)
    with pytest.raises(RedisReplyError):
        client.rename_tensor("not-a-tensor", "still-not-a-tensor")


def test_copy_nonexistant_key(context):

    client = Client(None, logger_name=context)
    with pytest.raises(RedisReplyError):
        client.copy_tensor("not-a-tensor", "still-not-a-tensor")


def test_copy_not_tensor(context):
    def test_func(param):
        print(param)

    client = Client(None, logger_name=context)
    client.set_function("test_func", test_func)
    with pytest.raises(RedisReplyError):
        client.copy_tensor("test_func", "test_fork")


# --------------- Helper Functions --------------------


def get_prefix():
    # get prefix, if it exists. Assumes the client is using
    # tensor prefix which is the default.
    sskeyin = os.environ.get("SSKEYIN", None)
    prefix = ""
    if sskeyin:
        prefix = sskeyin.split(",")[0] + "."
    return prefix
