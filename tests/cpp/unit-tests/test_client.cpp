/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "../../../third-party/catch/catch.hpp"
#include "client.h"
#include "dataset.h"
#include "../client_test_utils.h"
#include "srexception.h"
#include <sstream>

using namespace SmartRedis;

// helper function that determines whether two
// vectors of type T contain the same elements
template<typename T>
bool is_same_data(T* t1, T* t2, size_t length)
{
    for(size_t i=0; i<length; i++) {
        if(*t1 != *t2)
            return false;
        t1++;
        t2++;
    }
    return true;
}

// helper function that returns the first address
// if SSDB contains more than one
std::string parse_SSDB(std::string addresses)
{
    std::stringstream address_stream(addresses);
    std::string first_address;
    getline(address_stream, first_address, ',');
    return first_address;
}

// auxiliary function for testing the equivalence
// of two vectors that each contain tensor data
void check_all_data(size_t length, std::vector<void*>& original_datas,
                    std::vector<void*>& new_datas)
{
    CHECK(is_same_data((double*)original_datas[0],
                       (double*)new_datas[0],
                       length));
    CHECK(is_same_data((float*)original_datas[1],
                       (float*)new_datas[1],
                       length));
    CHECK(is_same_data((int64_t*)original_datas[2],
                       (int64_t*)new_datas[2],
                       length));
    CHECK(is_same_data((int32_t*)original_datas[3],
                       (int32_t*)new_datas[3],
                       length));
    CHECK(is_same_data((int16_t*)original_datas[4],
                       (int16_t*)new_datas[4],
                       length));
    CHECK(is_same_data((int8_t*)original_datas[5],
                       (int8_t*)new_datas[5],
                       length));
    CHECK(is_same_data((uint16_t*)original_datas[6],
                       (uint16_t*)new_datas[6],
                       length));
    CHECK(is_same_data((uint8_t*)original_datas[7],
                       (uint8_t*)new_datas[7],
                       length));
}

SCENARIO("Testing Dataset Functions on Client Object", "[Client]")
{

    GIVEN("A Client object")
    {
        Client client(use_cluster());

        THEN("get, rename, and copy DataSet called on "
             "a nonexistent DataSet throws errors")
        {
            CHECK_THROWS_AS(
                client.get_dataset("DNE"),
                _smart_runtime_error);
            CHECK_THROWS_AS(
                client.rename_dataset("DNE", "rename_DNE"),
               _smart_runtime_error);
            CHECK_THROWS_AS(
                client.copy_dataset("src_DNE", "dest_DNE"),
                _smart_runtime_error);
        }

        WHEN("A dataset is created and put into the Client")
        {
            // Create the DataSet
            std::string dataset_name = "test_dataset_name";
            DataSet dataset(dataset_name);

            // Add meta scalar to DataSet
            std::string meta_scalar_name = "dbl_field";
            const double dbl_meta = std::numeric_limits<double>::max();
            SRMetaDataType meta_type = sr_meta_dbl;
            dataset.add_meta_scalar(meta_scalar_name,
                                     &dbl_meta,
                                     meta_type);

            // Add tensor to DataSet
            std::string tensor_name = "test_tensor";
            std::vector<size_t> dims = {1, 2, 3};
            SRTensorType type = sr_tensor_flt;
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<float> tensor(tensor_size, 2.0);
            void* data = tensor.data();
            SRMemoryLayout mem_layout = sr_layout_contiguous;
            dataset.add_tensor(tensor_name, data, dims, type, mem_layout);

            // Put the DataSet into the Client
            client.put_dataset(dataset);

            THEN("The DataSet can be retrieved")
            {
                // Get the DataSet
                double* retrieved_meta_data;
                size_t retrieved_meta_length;
                DataSet retrieved_dataset = client.get_dataset(dataset_name);

                // Ensure the DataSet has the correct data
                retrieved_dataset.get_meta_scalars(meta_scalar_name,
                                                  (void*&)retrieved_meta_data,
                                                   retrieved_meta_length,
                                                   meta_type);
                CHECK(*retrieved_meta_data == dbl_meta);
                CHECK(dataset.get_tensor_names() ==
                      retrieved_dataset.get_tensor_names());
            }

            AND_THEN("The DataSet can be renamed. (rename_dataset "
                     "calls copy and delete)")
            {
                // rename dataset
                std::string renamed_dataset_name = "renamed_" + dataset_name;
                client.rename_dataset(dataset_name, renamed_dataset_name);

                // original name no longer exists after renaming
                CHECK_THROWS_AS(
                    client.get_dataset(dataset_name),
                    _smart_runtime_error);

                // the dataset with new name can be retrieved
                double* retrieved_meta_data;
                size_t retrieved_meta_length;
                DataSet retrieved_dataset =
                    client.get_dataset(renamed_dataset_name);

                // ensure the retrieved dataset has the correct data
                retrieved_dataset.get_meta_scalars(meta_scalar_name,
                                                  (void*&)retrieved_meta_data,
                                                   retrieved_meta_length,
                                                   meta_type);
                CHECK(*retrieved_meta_data == dbl_meta);
                CHECK(dataset.get_tensor_names() ==
                      retrieved_dataset.get_tensor_names());
            }
        }
    }
}

SCENARIO("Testing Tensor Functions on Client Object", "[Client]")
{

    GIVEN("A Client object")
    {
        Client client(use_cluster());

        AND_WHEN("Tensors of each type are created and put into the Client")
        {
            SRMemoryLayout mem_layout = sr_layout_contiguous;
            const int num_of_tensors = 8;
            std::vector<std::vector<size_t> > dims(num_of_tensors, {2, 1});
            size_t tensors_size = dims[0][0] * dims[0][1];
            std::vector<std::string> keys = { "dbl_key", "flt_key",
                                              "int64_key", "int32_key",
                                              "int16_key", "int8_key",
                                              "uint16_key", "uint8_key"};
            std::vector<SRTensorType> types =
                {sr_tensor_dbl, sr_tensor_flt,
                 sr_tensor_int64, sr_tensor_int32,
                 sr_tensor_int16, sr_tensor_int8,
                 sr_tensor_uint16, sr_tensor_uint8};

            std::vector<double> dbl_tensor =
                {std::numeric_limits<double>::min(),
                 std::numeric_limits<double>::max()};
            std::vector<float> flt_tensor =
                {std::numeric_limits<float>::min(),
                 std::numeric_limits<float>::max()};
            std::vector<int64_t> int64_tensor =
                {std::numeric_limits<int64_t>::min(),
                 std::numeric_limits<int64_t>::max()};
            std::vector<int32_t> int32_tensor =
                {std::numeric_limits<int32_t>::min(),
                 std::numeric_limits<int32_t>::max()};
            std::vector<int16_t> int16_tensor =
                {std::numeric_limits<int16_t>::min(),
                 std::numeric_limits<int16_t>::max()};
            std::vector<int8_t> int8_tensor =
                {std::numeric_limits<int8_t>::min(),
                 std::numeric_limits<int8_t>::max()};
            std::vector<uint16_t> uint16_tensor =
                {std::numeric_limits<uint16_t>::min(),
                 std::numeric_limits<uint16_t>::max()};
            std::vector<uint8_t> uint8_tensor =
                {std::numeric_limits<uint8_t>::min(),
                 std::numeric_limits<uint8_t>::max()};

            std::vector<void*> datas =
                {dbl_tensor.data(), flt_tensor.data(),
                 int64_tensor.data(), int32_tensor.data(),
                 int16_tensor.data(), int8_tensor.data(),
                 uint16_tensor.data(), uint8_tensor.data()};

            // put each tensor into the Client
            for (int i=0; i<num_of_tensors; i++)
                client.put_tensor(keys[i], datas[i], dims[i],
                                  types[i], mem_layout);

            THEN("The Tensors can be retrieved")
            {
                SRTensorType retrieved_type;
                std::vector<void*> retrieved_datas(num_of_tensors);
                std::vector<std::vector<size_t>>
                    retrieved_dims(num_of_tensors);

                // Get the tensors and ensure the correct data is retrieved
                for(int i=0; i<num_of_tensors; i++) {
                    client.get_tensor(keys[i], retrieved_datas[i],
                                      retrieved_dims[i], retrieved_type,
                                      mem_layout);
                    CHECK(retrieved_dims[i] == dims[i]);
                    CHECK(retrieved_type == types[i]);
                }
                check_all_data(tensors_size, datas, retrieved_datas);
            }

            AND_THEN("The Tensors can be retrieved with the c-style interface")
            {
                std::vector<void*> retrieved_datas(num_of_tensors);
                std::vector<size_t*> retrieved_dims(num_of_tensors);
                std::vector<size_t> retrieved_n_dims(num_of_tensors);
                SRTensorType retrieved_type;

                // Get the tensors and ensure the correct data is retrieved
                for(int i=0; i<num_of_tensors; i++) {
                    client.get_tensor(keys[i], retrieved_datas[i],
                                      retrieved_dims[i], retrieved_n_dims[i],
                                      retrieved_type, mem_layout);

                    CHECK(retrieved_type == types[i]);
                    CHECK(retrieved_n_dims[i] == dims[i].size());
                    size_t* retrieved_dim = retrieved_dims[i];
                    for(size_t j=0; j<retrieved_n_dims[i]; j++) {
                        CHECK(*retrieved_dim == dims[i][j]);
                        retrieved_dim++;
                    }
                }
                check_all_data(tensors_size, datas, retrieved_datas);
            }

            AND_THEN("The Tensors can be unpacked")
            {
                // allocate memory for the tensors that will be unpacked
                std::vector<void*> retrieved_datas = {
                    malloc(dims[0][0] * dims[0][1] * sizeof(double)),
                    malloc(dims[1][0] * dims[1][1] * sizeof(float)),
                    malloc(dims[2][0] * dims[2][1] * sizeof(int64_t)),
                    malloc(dims[3][0] * dims[3][1] * sizeof(int32_t)),
                    malloc(dims[4][0] * dims[4][1] * sizeof(int16_t)),
                    malloc(dims[5][0] * dims[5][1] * sizeof(int8_t)),
                    malloc(dims[6][0] * dims[6][1] * sizeof(uint16_t)),
                    malloc(dims[7][0] * dims[7][1] * sizeof(uint8_t))};

                // unpack each tensor into retrieved_datas
                for(int i=0; i<num_of_tensors; i++) {
                    client.unpack_tensor(keys[i], retrieved_datas[i],
                                        {dims[i][0]*dims[i][1]}, types[i],
                                         mem_layout);
                }
                check_all_data(tensors_size, datas, retrieved_datas);

                for(int i=0; i<retrieved_datas.size(); i++)
                    free(retrieved_datas[i]);
            }

            AND_THEN("A contiguous tensor "
                     "can be unpacked incorrectly")
            {
                // allocate memory for the unpack_tensor calls
                void* contig_retrieved_data =
                    malloc(dims[0][0] * dims[0][1] * sizeof(double));

                // incorrectly unpack contiguous tensor (contiguous mem space doesn't
                // match the dimensions that were fetched)
                CHECK_THROWS_AS(
                    client.unpack_tensor(keys[0], contig_retrieved_data,
                                         dims[0], sr_tensor_dbl,
                                         sr_layout_contiguous),
                    _smart_runtime_error);

                free(contig_retrieved_data);

                // TODO: Do this same test again, but for nested MemoryLayout
            }

            AND_THEN("The Tensors can be renamed")
            {
                for(int i=0; i<num_of_tensors; i++) {
                    client.rename_tensor(keys[i], "renamed_" + keys[i]);
                    // Ensure the tensor with old name doesnt exist anymore
                    CHECK_FALSE(client.tensor_exists(keys[i]));
                    // Ensure the tensor with the new name exists
                    CHECK(client.tensor_exists("renamed_" + keys[i]) == true);
                }

                // Ensure the tensors were correctly migrated to their new name
                SRTensorType retrieved_type;
                std::vector<void*> retrieved_datas(num_of_tensors);
                std::vector<std::vector<size_t>>
                    retrieved_dims(num_of_tensors);

                for(int i=0; i<num_of_tensors; i++) {
                    std::string renamed_key = "renamed_" + keys[i];
                    client.get_tensor(renamed_key, retrieved_datas[i],
                                      retrieved_dims[i], retrieved_type,
                                      mem_layout);
                    CHECK(retrieved_dims[i] == dims[i]);
                    CHECK(retrieved_type == types[i]);
                }
                check_all_data(tensors_size, datas, retrieved_datas);
            }

            AND_THEN("A Tensor can be renamed to the same name")
            {
                client.rename_tensor(keys[0], keys[0]);
                CHECK(client.tensor_exists(keys[0]) == true);
            }

            AND_THEN("The Tensors can be deleted")
            {
                for(int i=0; i<num_of_tensors; i++) {
                    client.delete_tensor(keys[i]);
                    CHECK_FALSE(client.tensor_exists(keys[i]));
                }
            }

            AND_THEN("The Tensors can be copied")
            {
                // copy each tensor
                for(int i=0; i<num_of_tensors; i++)
                    client.copy_tensor(keys[i], "copied_" + keys[i]);

                // ensure the copied tensors contain
                // the correct data, dims, type
                SRTensorType copied_type;
                std::vector<void*> copied_datas(num_of_tensors);
                std::vector<std::vector<size_t>> copied_dims(num_of_tensors);

                for(int i=0; i<num_of_tensors; i++) {
                    client.get_tensor("copied_" + keys[i], copied_datas[i],
                                      copied_dims[i], copied_type,
                                      mem_layout);
                    CHECK(copied_dims[i] == dims[i]);
                    CHECK(copied_type == types[i]);
                }
                check_all_data(tensors_size, datas, copied_datas);

                // ensure the original tensors' states are preserved
                // when the copied tensors are deleted
                for(int i=0; i<num_of_tensors; i++)
                    client.delete_tensor("copied_" + keys[i]);

                SRTensorType retrieved_type;
                std::vector<void*> retrieved_datas(num_of_tensors);
                std::vector<std::vector<size_t>>
                    retrieved_dims(num_of_tensors);

                for(int i=0; i<num_of_tensors; i++) {
                    client.get_tensor(keys[i], retrieved_datas[i],
                                      retrieved_dims[i], retrieved_type,
                                      mem_layout);
                    CHECK(retrieved_dims[i] == dims[i]);
                    CHECK(retrieved_type == types[i]);
                }
                check_all_data(tensors_size, datas, retrieved_datas);
            }

            AND_THEN("The Tensors can be polled")
            {
                int poll_freq = 10;
                int num_tries = 4;
                // polling tensors that exist returns true
                for(int i=0; i<num_of_tensors; i++)
                    CHECK(true == client.poll_tensor(keys[i],
                                                     poll_freq,
                                                     num_tries));

                // polling a tensor that does not exist returns false
                CHECK_FALSE(client.poll_tensor("DNE",
                                                poll_freq,
                                                num_tries));
            }

            AND_THEN("The keys can be polled")
            {
                int poll_freq = 10;
                int num_tries = 4;
                std::string prefix = get_prefix();
                // polling keys that exist returns true
                for(int i=0; i<num_of_tensors; i++) {
                    CHECK(true == client.poll_key(prefix + keys[i],
                                                  poll_freq,
                                                  num_tries));
                }

                // polling a key that does not exist returns false
                CHECK_FALSE(client.poll_key("DNE",
                                             poll_freq,
                                             num_tries));
            }

            AND_THEN("A Tensor can be incorrectly unpacked, resulting in a runtime error")
            {
                // contiguous and total_dims != dims[0] throws error in unpack_tensor
                void* retrieved_data =
                    malloc(dims[0][0] * dims[0][1] * sizeof(double));
                std::vector<size_t> incorrect_dims = {10};
                CHECK_THROWS_AS(
                    client.unpack_tensor(keys[0], retrieved_data,
                                         incorrect_dims, types[0],
                                         mem_layout),
                    _smart_runtime_error);

                free(retrieved_data);
            }
        }
    }
}

SCENARIO("Testing INFO Functions on Client Object", "[Client]")
{

    GIVEN("A Client object")
    {
        Client client(use_cluster());

        WHEN("INFO or CLUSTER INFO is called on database with "
             "an invalid address")
        {
            THEN("An error is thrown")
            {
                std::string db_address = ":00";

                CHECK_THROWS_AS(client.get_db_node_info(db_address),
                                _smart_runtime_error);
                CHECK_THROWS_AS(client.get_db_cluster_info(db_address),
                                _smart_runtime_error);
            }
        }

        AND_WHEN("INFO is called on database with a valid address ")
        {

            THEN("No errors with be thrown for both cluster and "
                 "non-cluster environemnts")
            {
                std::string db_address = parse_SSDB(std::getenv("SSDB"));

                CHECK_NOTHROW(client.get_db_node_info(db_address));
            }
        }

        AND_WHEN("CLUSTER INFO is called with a valid address ")
        {
            THEN("No errors are thrown if called on a cluster environment "
                 "but errors are thrown if called on a non-cluster environment")
            {
                std::string db_address = parse_SSDB(std::getenv("SSDB"));
                if (use_cluster())
                    CHECK_NOTHROW(client.get_db_cluster_info(db_address));
                else
                    CHECK_THROWS_AS(client.get_db_cluster_info(db_address),
                                    _smart_runtime_error);
            }
        }
    }
}

SCENARIO("Testing FLUSHDB on empty Client Object", "[Client][FLUSHDB]")
{

    GIVEN("An empty non-cluster Client object")
    {
        Client client(use_cluster());

        WHEN("FLUSHDB is called on database with "
             "an invalid address")
        {
            THEN("An error is thrown")
            {
                std::string db_address = ":00";

                CHECK_THROWS_AS(client.flush_db(db_address),
                                _smart_runtime_error);

                CHECK_THROWS_AS(client.flush_db("123456678.345633.21:2345561"),
                                _smart_runtime_error);
            }
        }

        AND_WHEN("FLUSHDB is called on database with "
                 "a valid address")
        {
            THEN("No errors are thrown")
            {
                std::string db_address = parse_SSDB(std::getenv("SSDB"));

                CHECK_NOTHROW(client.flush_db(db_address));
            }
        }
    }
}

SCENARIO("Testing FLUSHDB on Client Object", "[Client][FLUSHDB]")
{

    GIVEN("A non-cluster Client object")
    {
        // From within the testing framework, there is no way of knowing
        // each db node that is being used, so skip if on cluster
        if (use_cluster())
            return;

        Client client(use_cluster());
        std::string dataset_name = "test_dataset_name";
        DataSet dataset(dataset_name);
        dataset.add_meta_string("meta_string_name", "meta_string_val");
        std::string tensor_key = "dbl_tensor";
        std::vector<double> tensor_dbl =
                {std::numeric_limits<double>::min(),
                 std::numeric_limits<double>::max()};
        client.put_dataset(dataset);
        client.put_tensor(tensor_key, (void*)tensor_dbl.data(), {2,1},
                          sr_tensor_dbl, sr_layout_contiguous);
        WHEN("FLUSHDB is called on databsase")
        {

            THEN("The database is flushed")
            {
                // ensure the database has things to flush
                CHECK(client.dataset_exists(dataset_name) == true);
                CHECK(client.tensor_exists(tensor_key) == true);
                // flush the database
                std::string db_address = parse_SSDB(std::getenv("SSDB"));
                CHECK_NOTHROW(client.flush_db(db_address));

                // ensure the database is empty
                CHECK_FALSE(client.dataset_exists(dataset_name));
                CHECK_FALSE(client.tensor_exists(tensor_key));
            }
        }
    }
}

SCENARIO("Testing CONFIG GET and CONFIG SET on Client Object", "[Client]")
{

    GIVEN("A Client object")
    {
        Client client(use_cluster());

        WHEN("CONFIG GET or CONFIG SET are called on databases with "
             "invalid addresses ")
        {
            THEN("An error is thrown")
            {
                std::vector<std::string> db_addresses =
                    {":00", "127.0.0.1:", "127.0.0.1", "127.0.0.1:18446744073709551616"};

                for (size_t address_index = 0; address_index < db_addresses.size(); address_index++) {
                    CHECK_THROWS_AS(client.config_get("*max-*-entries*", db_addresses[address_index]),
                                    _smart_runtime_error);
                    CHECK_THROWS_AS(client.config_set("dbfilename", "new_file.rdb", db_addresses[address_index]),
                                    _smart_runtime_error);
                }
            }
        }

        AND_WHEN("CONFIG GET or CONFIG SET are called on databases with "
                 "valid addresses ")
        {
            THEN("No error is thrown."){

                std::string db_address = parse_SSDB(std::getenv("SSDB"));
                std::string config_param = "dbfilename";
                std::string new_filename = "new_file.rdb";

                CHECK_NOTHROW(client.config_set(config_param, new_filename, db_address));
                std::unordered_map<std::string,std::string> reply =
                    client.config_get("dbfilename", db_address);

                CHECK(reply.size() == 1);
                REQUIRE(reply.count(config_param) > 0);
                CHECK(reply[config_param] == new_filename);
            }
        }
    }
}

SCENARIO("Test CONFIG GET on an unsupported command", "[Client]")
{
    GIVEN("A client object")
    {
        Client client(use_cluster());
        std::string address = parse_SSDB(std::getenv("SSDB"));

        WHEN("CONFIG GET is called with an unsupported command")
        {
            std::unordered_map<std::string,std::string> reply =
                client.config_get("unsupported_cmd", address);

            THEN("CONFIG GET returns an empty unordered map")
            {
                CHECK(reply.empty() == true);
            }
        }
    }
}

SCENARIO("Test CONFIG SET on an unsupported command", "[Client]")
{
    GIVEN("A client object")
    {
        Client client(use_cluster());
        std::string address = parse_SSDB(std::getenv("SSDB"));

        WHEN("CONFIG SET is called with an unsupported command")
        {

            THEN("CONFIG SET throws a runtime error")
            {
                CHECK_THROWS_AS(
                    client.config_set("unsupported_cmd", "100", address),
                    _smart_runtime_error);
            }
        }
    }
}

SCENARIO("Testing SAVE command on Client Object", "[Client][SAVE]")
{

    GIVEN("A client object and some data")
    {
        Client client(use_cluster());
        std::string dataset_name = "test_save_dataset";
        DataSet dataset(dataset_name);
        dataset.add_meta_string("meta_string_save_name", "meta_string_val");
        std::string tensor_key = "save_dbl_tensor";
        std::vector<double> tensor_dbl =
                {std::numeric_limits<double>::min(),
                 std::numeric_limits<double>::max()};
        client.put_dataset(dataset);
        client.put_tensor(tensor_key, (void*)tensor_dbl.data(), {2,1},
                          sr_tensor_dbl, sr_layout_contiguous);

        std::string address = parse_SSDB(std::getenv("SSDB"));

        WHEN("When SAVE is called for a given address")
        {

            THEN("Producing a point in time snapshot of the redis instance is successful")
            {
                // get the timestamp of the last SAVE
                parsed_reply_nested_map db_node_info_before = client.get_db_node_info(address);
                std::string time_before_save = db_node_info_before["Persistence"]["rdb_last_save_time"];

                CHECK_NOTHROW(client.save(address));

                // check that the timestamp of the last SAVE has increased
                parsed_reply_nested_map db_node_info_after = client.get_db_node_info(address);
                std::string time_after_save = db_node_info_after["Persistence"]["rdb_last_save_time"];

                CHECK(time_before_save.compare(time_after_save) < 0);
            }
        }
    }
}

SCENARIO("Test that prefixing covers all hash slots of a cluster", "[Client]")
{

    if(use_cluster()==false)
        return;

    GIVEN("A test RedisCluster test object")
    {
        RedisClusterTestObject redis_cluster;

        WHEN("A prefix is requested for a hash slot between 0 and 16384")
        {
            for(size_t hash_slot = 0; hash_slot <= 16384; hash_slot++) {

                THEN("'{' and '}' do not appear in the prefix")
                {
                    std::string prefix = redis_cluster.get_crc16_prefix(hash_slot);
                    CHECK(prefix.size() > 0);
                    CHECK(prefix.find('{') == std::string::npos);
                    CHECK(prefix.find('}') == std::string::npos);
                    size_t redispp_hash_slot =
                        sw::redis::crc16(prefix.c_str(), prefix.size())%16384;
                    CHECK(hash_slot == redispp_hash_slot);
                }
            }
        }

        WHEN("A prefix is requested for a hash slot out of range")
        {
            THEN("A std::runtime_error is thrown")
            {
                CHECK_THROWS_AS(redis_cluster.get_crc16_prefix(16385),
                                _smart_runtime_error);
            }
        }

    }
}
