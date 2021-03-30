#include "client.h"
#include <vector>
#include <string>

int main(int argc, char* argv[]) {

    // Initialize tensor dimensions
    size_t dim1 = 3;
    size_t dim2 = 2;
    size_t dim3 = 5;
    size_t n_values = dim1 * dim2 * dim3;
    std::vector<size_t> dims = {3, 2, 5};

    // Initialize two tensors to random values
    std::vector<double> tensor_1(n_values, 0);
    std::vector<int64_t> tensor_2(n_values, 0);

    for(size_t i=0; i<n_values; i++) {
        tensor_1[i] = 2.0*rand()/RAND_MAX - 1.0;
        tensor_2[i] = rand();
    }

    // Initialize three metadata values we will add
    // to the DataSet
    uint32_t meta_scalar_1 = 1;
    uint32_t meta_scalar_2 = 2;
    int64_t meta_scalar_3 = 3;

    // Initialize a SmartRedis client
    SmartRedis::Client client(false);

    // Create a DataSet
    SmartRedis::DataSet dataset("example_dataset");

    // Add tensors to the DataSet
    dataset.add_tensor("tensor_1", tensor_1.data(), dims,
                       SmartRedis::TensorType::dbl,
                       SmartRedis::MemoryLayout::contiguous);

    dataset.add_tensor("tensor_2", tensor_2.data(), dims,
                       SmartRedis::TensorType::int64,
                       SmartRedis::MemoryLayout::contiguous);

    // Add metadata scalar values to the DataSet
    dataset.add_meta_scalar("meta_field_1", &meta_scalar_1,
                            SmartRedis::MetaDataType::uint32);
    dataset.add_meta_scalar("meta_field_1", &meta_scalar_2,
                            SmartRedis::MetaDataType::uint32);
    dataset.add_meta_scalar("meta_field_2", &meta_scalar_3,
                            SmartRedis::MetaDataType::int64);


    // Put the DataSet in the database
    client.put_dataset(dataset);

    // Retrieve the DataSet from the database
    SmartRedis::DataSet retrieved_dataset =
        client.get_dataset("example_dataset");

    // Retrieve one of the tensors
    std::vector<int64_t> unpack_dataset_tensor(n_values, 0);
    retrieved_dataset.unpack_tensor("tensor_2",
                                    unpack_dataset_tensor.data(),
                                    {n_values},
                                    SmartRedis::TensorType::int64,
                                    SmartRedis::MemoryLayout::contiguous);

    // Print out the retrieved values
    std::cout<<"Comparing sent and received "\
               "values for tensor_2: "<<std::endl;

    for(size_t i=0; i<n_values; i++)
        std::cout<<"Sent: "<<tensor_2[i]<<" "
                 <<"Received: "
                 <<unpack_dataset_tensor[i]<<std::endl;

    //Retrieve a metadata field
    size_t get_n_meta_values;
    void* get_meta_values;
    SmartRedis::MetaDataType get_type;
    dataset.get_meta_scalars("meta_field_1",
                             get_meta_values,
                             get_n_meta_values,
                             get_type);

    // Print out the metadata field values
    for(size_t i=0; i<get_n_meta_values; i++)
        std::cout<<"meta_field_1 value "<<i<<" = "
                 <<((uint32_t*)get_meta_values)[i]<<std::endl;

    return 0;
}
