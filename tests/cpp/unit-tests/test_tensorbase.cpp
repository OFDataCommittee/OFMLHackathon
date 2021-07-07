#include "catch.hpp"
#include "tensorbase.h"
#include "tensorpack.h"

using namespace SmartRedis;

SCENARIO("Testing TensorBase through TensorList", "[TensorBase]")
{
    TensorType tensor_type = GENERATE(TensorType::dbl, TensorType::flt, TensorType::int64,
                                      TensorType::int32, TensorType::int16, TensorType::int8,
                                      TensorType::uint16, TensorType::uint8);
    GIVEN("A TensorPack object")
    {
        TensorPack tp;
        WHEN("A valid tensor is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {1, 2, 3};
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<double> tensor(tensor_size, 0);
            for (size_t i=0; i<tensor_size; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();
            tp.add_tensor(name, data, dims,
                          tensor_type,
                          MemoryLayout::contiguous);
            THEN("The tensor and its data members can be retrieved")
            {
                TensorBase* tensor_ptr = tp.get_tensor(name);
                CHECK(tensor_ptr->name() == name);
                CHECK(tensor_ptr->type() == tensor_type);
                CHECK(tensor_ptr->type_str() == TENSOR_STR_MAP.at(tensor_type));
                CHECK(tensor_ptr->dims() == dims);
                CHECK(tensor_ptr->num_values() == tensor_size);
                tensor_ptr->data_view(MemoryLayout::contiguous);
                // TODO: Check that tensor_ptr's data() and buf() methods return correct value
            }
            AND_THEN("The tensor can be copied")
            {
                TensorPack tp_cpy(tp);
                TensorBase* tensor_ptr = tp.get_tensor(name);
                TensorBase* tensor_cpy_ptr = tp_cpy.get_tensor(name);
                CHECK(tensor_ptr->name() == tensor_cpy_ptr->name());
                CHECK(tensor_ptr->type() == tensor_cpy_ptr->type());
                CHECK(tensor_ptr->type_str() == tensor_cpy_ptr->type_str());
                CHECK(tensor_ptr->dims() == tensor_cpy_ptr->dims());
                CHECK(tensor_ptr->num_values() == tensor_cpy_ptr->num_values());
                CHECK(tensor_ptr->buf() == tensor_cpy_ptr->buf());
            }
        }
        AND_WHEN("A tensor with data as nullptr is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {1, 1, 1};
            std::vector<double> tensor;
            void* data = tensor.data();
            THEN("A runtime error is thrown during TensorType instantiation")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                          tensor_type,
                          MemoryLayout::contiguous),
                    std::runtime_error
                );
            }
        }
        AND_WHEN("A tensor with an empty string as its name is added to the tensorpack")
        {
            std::string name = "";
            std::vector<size_t> dims = {1, 2, 3};
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<double> tensor(tensor_size, 0);
            for (size_t i=0; i<tensor_size; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();
            THEN("A runtime error is thrown during TensorType instantiation")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                          tensor_type,
                          MemoryLayout::contiguous),
                    std::runtime_error
                );
            }
        }
        AND_WHEN("A tensor with '.meta' as its name is added to the tensorpack")
        {
            std::string name = ".meta";
            std::vector<size_t> dims = {1, 2, 3};
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<double> tensor(tensor_size, 0);
            for (size_t i=0; i<tensor_size; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();
            THEN("A runtime error is thrown during TensorType instantiation")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                          tensor_type,
                          MemoryLayout::contiguous),
                    std::runtime_error
                );
            }
        }
        AND_WHEN("A tensor with zero dimensions is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {};
            std::vector<double> tensor(5, 0);
            for (size_t i=0; i<5; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();
            THEN("A runtime error is thrown during TensorType instantiation")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                          tensor_type,
                          MemoryLayout::contiguous),
                    std::runtime_error
                );
            }
        }
        AND_WHEN("A tensor with a dimension that is <= 0 is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {1, 0, 3};
            std::vector<double> tensor(5, 0);
            for (size_t i=0; i<5; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();
            THEN("A runtime error is thrown during TensorType instantiation")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                                  tensor_type, MemoryLayout::contiguous),
                    std::runtime_error);
            }
        }
    }
}