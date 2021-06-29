#include "catch.hpp"
#include "tensor.h"

using namespace SmartRedis;

SCENARIO("Testing Tensor", "[Tensor]")
{
    GIVEN("Two Tensors")
    {
        std::string name = "test_tensor";
        std::vector<size_t> dims = {1, 2, 3};
        TensorType type = TensorType::flt;
        size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
        std::vector<float> tensor(tensor_size, 0);
        for (size_t i=0; i<tensor_size; i++)
            tensor[i] = 2.0*rand()/RAND_MAX -1.0;
        void* data = tensor.data();
        MemoryLayout mem_layout = MemoryLayout::contiguous;
        //Tensor<float>* t = new Tensor<float>(name, data, dims, type, mem_layout);
        Tensor<float> t(name, data, dims, type, mem_layout);

        std::string name_2 = "test_tensor_2";
        std::vector<size_t> dims_2 = {3, 2, 2};
        TensorType type_2 = TensorType::flt;
        size_t tensor_size_2 = dims_2.at(0) * dims_2.at(1) * dims_2.at(2);
        std::vector<float> tensor_2(tensor_size_2, 0);
        for (size_t i=0; i<tensor_size_2; i++)
            tensor_2[i] = 2.0*rand()/RAND_MAX -1.0;
        void* data_2 = tensor_2.data();
        MemoryLayout mem_layout_2 = MemoryLayout::contiguous;
        Tensor<float> t_2(name_2, data_2, dims_2, type_2, mem_layout_2);
        WHEN("A tensor is copied with the assignment operator")
        {
            // t_2 = t;
            THEN("The two tensors are the same")
            {
                // CHECK(t->name() == t_2.name());
                // CHECK(t->type() == t_2.type());
                // CHECK(t->type_str() == t_2.type_str());
                // CHECK(t->dims() == t_2.dims());
                // CHECK(t->num_values() == t_2.num_values());
                // CHECK(*((float*)t->data()) == *((float*)t_2.data()));
                // CHECK(t->buf() == t_2.buf());
            }
        }
        AND_WHEN("A tensor is constructed with the move constructor")
        {

            void* prev_addr = t_2.data();
            t_2 = std::move(t);
            void* post_addr = t_2.data();
            CHECK(prev_addr != post_addr);
            CHECK(name == t_2.name());
            CHECK(type == t_2.type());
            CHECK(dims == t_2.dims());


        }
        // delete t;
    }
}