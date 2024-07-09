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
from smartredis import ConfigOptions
from smartredis.error import *


def test_dbnode_info_command(context):
    ssdb = os.environ["SSDB"]
    addresses = ssdb.split(',')
    client = Client(None, logger_name=context)
    info = client.get_db_node_info(addresses)
    assert len(info) > 0

def test_dbcluster_info_command(mock_model, context):
    ssdb = os.environ["SSDB"]
    addresses = ssdb.split(',')
    co = ConfigOptions().create_from_environment("")
    client = Client(co, logger_name=context)

    if os.environ["SR_DB_TYPE"] == "Clustered":
        info = client.get_db_cluster_info(addresses)
        assert len(info) > 0
    else:
        # cannot call get_db_cluster_info in non-cluster environment
        with pytest.raises(RedisReplyError):
            client.get_db_cluster_info(addresses)

    # Get a mock model
    model = mock_model.create_torch_cnn()

    # set the mock model
    client.set_model("ai_info_cnn", model, "TORCH", "CPU")

    # Check with valid address and model key
    ai_info = client.get_ai_info(addresses, "ai_info_cnn")
    assert len(ai_info) != 0

    # Check that invalid address throws error
    with pytest.raises(RedisRuntimeError):
        client.get_ai_info(["no_host:6379"], "ai_info_cnn")

    # Check that invalid model name throws error
    with pytest.raises(RedisRuntimeError):
        client.get_ai_info(addresses, "bad_key")

def test_flushdb_command(context):
    # from within the testing framework, there is no way
    # of knowing each db node that is being used, so skip
    # if on cluster
    ssdb = os.environ["SSDB"]
    addresses = ssdb.split(',')
    if os.environ["SR_DB_TYPE"] == "Clustered":
        return

    client = Client(None, logger_name=context)

    # add key to client via put_tensor
    tensor = np.array([1, 2])
    client.put_tensor("test_copy", tensor)

    assert client.tensor_exists("test_copy")
    client.flush_db(addresses)
    assert not client.tensor_exists("test_copy")


def test_config_set_get_command(context):
    # get env var to set through client init
    ssdb = os.environ["SSDB"]
    client = Client(None, logger_name=context)

    value = "6000"
    client.config_set("lua-time-limit", value, ssdb)
    get_reply = client.config_get("lua-time-limit", ssdb)
    assert len(get_reply) > 0
    assert get_reply["lua-time-limit"] == value


def test_config_set_command_DNE(context):
    ssdb = os.environ["SSDB"]
    client = Client(None, logger_name=context)

    # The CONFIG parameter "config_param_DNE" is unsupported
    with pytest.raises(RedisReplyError):
        client.config_set("config_param_DNE", "10", ssdb)


def test_config_get_command_DNE(context):
    ssdb = os.environ["SSDB"]
    client = Client(None, logger_name=context)

    # CONFIG GET returns an empty dictionary if the config_param is unsupported
    get_reply = client.config_get("config_param_DNE", ssdb)
    assert get_reply == dict()


def test_save_command(context):
    ssdb = os.environ["SSDB"]
    client = Client(None, logger_name=context)

    addresses = ssdb.split(",")

    # for each address, check that the timestamp of the last SAVE increases after calling Client::save
    for address in addresses:
        save_time_before = client.get_db_node_info([address])[0]["Persistence"]["rdb_last_save_time"]
        client.save([address])
        save_time_after = client.get_db_node_info([address])[0]["Persistence"]["rdb_last_save_time"]

        assert save_time_before <= save_time_after
