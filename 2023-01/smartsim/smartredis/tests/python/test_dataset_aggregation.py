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
# SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from smartredis import Client, Dataset
from smartredis.error import *
from smartredis import *


def test_aggregation(context):
    num_datasets = 4
    client = Client(None, logger_name=context)
    log_data(context, LLDebug, "Initialization complete")

    # Build datasets
    original_datasets = [create_dataset(f"dataset_{i}") for i in range(num_datasets)]
    log_data(context, LLDebug, "DataSets built")

    # Make sure the list is cleared
    list_name = "dataset_test_list"
    client.delete_list(list_name)
    log_data(context, LLDebug, "list cleared")

    # Put datasets into the list
    for i in range(num_datasets):
        client.put_dataset(original_datasets[i])
        client.append_to_list(list_name, original_datasets[i])
    log_data(context, LLDebug, "DataSets added to list")

    # Confirm that poll for list length works correctly
    actual_length = num_datasets
    poll_result = client.poll_list_length(list_name, actual_length, 100, 5)
    if (poll_result == False):
        raise RuntimeError(
            f"Polling for list length of {actual_length} returned "
            f"False for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 1")

    poll_result = client.poll_list_length(list_name, actual_length + 1, 100, 5)
    if (poll_result == True):
        raise RuntimeError(
            f"Polling for list length of {actual_length + 1} returned "
            f"True for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 2")

    # Confirm that poll for greater than or equal list length works correctly
    poll_result = client.poll_list_length_gte(list_name, actual_length - 1, 100, 5)
    if (poll_result == False):
        raise RuntimeError(
            f"Polling for list length greater than or equal to {actual_length - 1} "
            f"returned False for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 3")

    poll_result = client.poll_list_length_gte(list_name, actual_length, 100, 5)
    if (poll_result == False):
        raise RuntimeError(
            f"Polling for list length greater than or equal to {actual_length} "
            f"returned False for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 4")

    poll_result = client.poll_list_length_gte(list_name, actual_length + 1, 100, 5)
    if (poll_result == True):
        raise RuntimeError(
            f"Polling for list length greater than or equal to {actual_length + 1} "
            f"returned True for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 5")

    # Confirm that poll for less than or equal list length works correctly
    poll_result = client.poll_list_length_lte(list_name, actual_length - 1, 100, 5)
    if (poll_result == True):
        raise RuntimeError(
            f"Polling for list length less than or equal to {actual_length - 1} "
            f"returned True for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 6")

    poll_result = client.poll_list_length_lte(list_name, actual_length, 100, 5)
    if (poll_result == False):
        raise RuntimeError(
            f"Polling for list length less than or equal to {actual_length} "
            f"returned False for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 7")

    poll_result = client.poll_list_length_lte(list_name, actual_length + 1, 100, 5)
    if (poll_result == False):
        raise RuntimeError(
            f"Polling for list length less than or equal to {actual_length + 1} "
            f"returned False for known length of {actual_length}.")
    log_data(context, LLDebug, "Polling 8")

    # Check the list length
    list_length = client.get_list_length(list_name)
    if (list_length != actual_length):
        raise RuntimeError(
            f"The list length of {list_length} does not match expected "
            f"value of {actual_length}.")
    log_data(context, LLDebug, "List length check")
    
    # Check the return of a range of datasets from the aggregated list
    num_datasets = client.get_dataset_list_range(list_name, 0, 1)
    if (len(num_datasets) != 2):
        raise RuntimeError(
            f"The length is {len(num_datasets)}, which does not "
            f"match expected value of 2.")
    log_data(context, LLDebug, "Retrieve datasets from list checked")

    # Retrieve datasets via the aggregation list
    datasets = client.get_datasets_from_list(list_name)
    if len(datasets) != list_length:
        raise RuntimeError(
            f"The number of datasets received {len(datasets)} "
            f"does not match expected value of {list_length}.")
    for ds in datasets:
        check_dataset(ds)
    log_data(context, LLDebug, "DataSet list retrieval")
    
    # Rename a list of datasets
    client.rename_list(list_name, "new_list_name")
    renamed_list_datasets = client.get_datasets_from_list("new_list_name")
    if len(renamed_list_datasets) != list_length:
        raise RuntimeError(
            f"The number of datasets received {len(datasets)} "
            f"does not match expected value of {list_length}.")
    for ds in renamed_list_datasets:
        check_dataset(ds)
    log_data(context, LLDebug, "DataSet list rename complete")
    
    # Copy a list of datasets
    client.copy_list("new_list_name", "copied_list_name")
    copied_list_datasets = client.get_datasets_from_list("copied_list_name")
    if len(copied_list_datasets) != list_length:
        raise RuntimeError(
            f"The number of datasets received {len(datasets)} "
            f"does not match expected value of {list_length}.")
    for ds in copied_list_datasets:
        check_dataset(ds)
    log_data(context, LLDebug, "DataSet list copied")
    
    

# ------------ helper functions ---------------------------------


def create_dataset(name):
    array = np.array([1, 2, 3, 4])
    string = "test_meta_strings"
    scalar = 7

    dataset = Dataset(name)
    dataset.add_tensor("test_array", array)
    dataset.add_meta_string("test_string", string)
    dataset.add_meta_scalar("test_scalar", scalar)
    return dataset

def check_dataset(ds):
    comp_array = np.array([1, 2, 3, 4])
    tensor_name = "test_array"
    comp_string = "test_meta_strings"
    string_name = "test_string"

    array = ds.get_tensor(tensor_name)
    np.testing.assert_array_equal(
        array, comp_array, "array in retrieved dataset is not correct"
    )

    string = ds.get_meta_strings(string_name)
    np.testing.assert_array_equal(
        string, comp_string, "string in retrieved dataset is not correct"
    )
