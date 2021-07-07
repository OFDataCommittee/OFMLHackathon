#include "catch.hpp"
#include "../client_test_utils.h"
#include "client.h"
#include "dataset.h"
// #include "client_test_utils.h"
// #include "dataset_test_utils.h"

using namespace SmartRedis;

// helper function that determines whether two vectors of type T contain the same elements
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

void check_all_data(std::vector<void*>& original_datas, std::vector<void*>& new_datas, size_t length)
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

SCENARIO("Testing Client Object", "[Client]")
{
    GIVEN("A Client object not connected to a redis cluster")
    {
        bool use_cluster = false;
        Client client(use_cluster);
        THEN("get, rename, copy, and delete DataSet called on a nonexistent DataSet throws errors")
        {
            CHECK_THROWS_AS(
                client.get_dataset("DNE"),
                std::runtime_error
            );
            CHECK_THROWS_AS(
                client.rename_dataset("DNE", "rename_DNE"),
                std::runtime_error
            );
            CHECK_THROWS_AS(
                client.copy_dataset("src_DNE", "dest_DNE"),
                std::runtime_error
            );
            CHECK_THROWS_AS(
                client.delete_dataset("DNE"),
                std::runtime_error
            );
        }
        WHEN("A dataset is created and put into the Client")
        {
            std::string dataset_name = "test_dataset_name";
            DataSet dataset(dataset_name);
            // Add meta scalar to DataSet
            std::string meta_scalar_name = "dbl_field";
            const double dbl_meta = std::numeric_limits<double>::max();
            MetaDataType meta_type = MetaDataType::dbl;
            dataset.add_meta_scalar(meta_scalar_name,
                                     &dbl_meta,
                                     meta_type);
            // Add tensor to DataSet
            std::string tensor_name = "test_tensor";
            std::vector<size_t> dims = {1, 2, 3};
            TensorType type = TensorType::flt;
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<float> tensor(tensor_size, 2.0);
            void* data = tensor.data();
            MemoryLayout mem_layout = MemoryLayout::contiguous;
            dataset.add_tensor(tensor_name, data, dims, type, mem_layout);
            // put the DataSet into the Client
            client.put_dataset(dataset);
            THEN("The DataSet can be retrieved")
            {
                double* retrieved_meta_data;
                size_t retrieved_meta_length;
                DataSet retrieved_dataset = client.get_dataset(dataset_name);
                retrieved_dataset.get_meta_scalars(meta_scalar_name,
                                                  (void*&)retrieved_meta_data,
                                                   retrieved_meta_length,
                                                   meta_type);
                CHECK(*retrieved_meta_data == dbl_meta);
                CHECK(dataset.get_tensor_names() == retrieved_dataset.get_tensor_names());
            }
            AND_THEN("The DataSet can be renamed. (rename_dataset calls copy and delete)")
            {
                std::string renamed_dataset_name = "renamed_" + dataset_name;
                client.rename_dataset(dataset_name, renamed_dataset_name);
                CHECK_THROWS_AS(
                    client.get_dataset(dataset_name),
                    std::runtime_error);
                double* retrieved_meta_data;
                size_t retrieved_meta_length;
                DataSet retrieved_dataset = client.get_dataset(renamed_dataset_name);
                retrieved_dataset.get_meta_scalars(meta_scalar_name,
                                                  (void*&)retrieved_meta_data,
                                                   retrieved_meta_length,
                                                   meta_type);
                CHECK(*retrieved_meta_data == dbl_meta);
                CHECK(dataset.get_tensor_names() == retrieved_dataset.get_tensor_names());
            }
        }
        AND_WHEN("Tensors are created and put into the DataSet")
        {
            MemoryLayout mem_layout = MemoryLayout::contiguous;
            const int num_of_tensors = 8;
            std::vector<std::vector<size_t> > dims(num_of_tensors, {2, 1});
            size_t tensors_size = dims[0][0] * dims[0][1];
            std::vector<std::string> keys = { "dbl_key", "flt_key",
                                              "int64_key", "int32_key",
                                              "int16_key", "int8_key",
                                              "uint16_key", "uint8_key"};
            std::vector<TensorType> types = {TensorType::dbl, TensorType::flt,
                                             TensorType::int64, TensorType::int32,
                                             TensorType::int16, TensorType::int8,
                                             TensorType::uint16, TensorType::uint8};
            std::vector<double> dbl_tensor = {std::numeric_limits<double>::min(),
                                              std::numeric_limits<double>::max()};
            std::vector<float> flt_tensor = {std::numeric_limits<float>::min(),
                                             std::numeric_limits<float>::max()};
            std::vector<int64_t> int64_tensor = {std::numeric_limits<int64_t>::min(),
                                                 std::numeric_limits<int64_t>::max()};
            std::vector<int32_t> int32_tensor = {std::numeric_limits<int32_t>::min(),
                                                 std::numeric_limits<int32_t>::max()};
            std::vector<int16_t> int16_tensor = {std::numeric_limits<int16_t>::min(),
                                                 std::numeric_limits<int16_t>::max()};
            std::vector<int8_t> int8_tensor = {std::numeric_limits<int8_t>::min(),
                                               std::numeric_limits<int8_t>::max()};
            std::vector<uint16_t> uint16_tensor = {std::numeric_limits<uint16_t>::min(),
                                                   std::numeric_limits<uint16_t>::max()};
            std::vector<uint8_t> uint8_tensor = {std::numeric_limits<uint8_t>::min(),
                                                 std::numeric_limits<uint8_t>::max()};
            std::vector<void*> datas = {dbl_tensor.data(), flt_tensor.data(),
                                        int64_tensor.data(), int32_tensor.data(),
                                        int16_tensor.data(), int8_tensor.data(),
                                        uint16_tensor.data(), uint8_tensor.data()};

            for (int i=0; i<num_of_tensors; i++)
                client.put_tensor(keys[i], datas[i], dims[i], types[i], mem_layout);
            THEN("The Tensors can be retrieved")
            {
                std::vector<void*> retrieved_datas(num_of_tensors);
                std::vector<std::vector<size_t> > retrieved_dims(num_of_tensors);
                TensorType retrieved_type;
                for(int i=0; i<num_of_tensors; i++) {
                    client.get_tensor(keys[i], retrieved_datas[i],
                                      retrieved_dims[i], retrieved_type,
                                      mem_layout);
                    CHECK(retrieved_dims[i] == dims[i]);
                    CHECK(retrieved_type == types[i]);
                }
                check_all_data(datas, retrieved_datas, tensors_size);
            }
            AND_THEN("The Tensors can be retrieved with the c-style interface")
            {
                std::vector<void*> retrieved_datas(num_of_tensors);
                std::vector<size_t*> retrieved_dims(num_of_tensors);
                std::vector<size_t> retrieved_n_dims(num_of_tensors);
                TensorType retrieved_type;
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
                check_all_data(datas, retrieved_datas, tensors_size);
            }
            AND_THEN("The Tensors can be unpacked")
            {
                // ...
            }
            AND_THEN("The Tensors can be renamed")
            {
                // ...
            }
            AND_THEN("The Tensors can be deleted")
            {
                // ...
            }
            AND_THEN("The Tensors can be copied")
            {
                // ...
            }
            AND_THEN("A Tensor can incorrectly be retrieved, resulting in a runtime error")
            {
                // dims.size <= 0
                // dims[i].size <= 0
                // TensorType::string
            }
        }
    }
}