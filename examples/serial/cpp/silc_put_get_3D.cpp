#include "client.h"
#include <vector>
#include <string>

int main(int argc, char* argv[]) {

    // Initialize tensor dimensions
    size_t dim1 = 3;
    size_t dim2 = 2;
    size_t dim3 = 5;
    std::vector<size_t> dims = {3, 2, 5};

    // Initialize a tensor to random values.  Note that a dynamically
    // allocated tensor via malloc is also useable with the client
    // API.  The std::vector is used here for brevity.
    size_t n_values = dim1 * dim2 * dim3;
    std::vector<double> input_tensor(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        input_tensor[i] = 2.0*rand()/RAND_MAX - 1.0;

    // Initialize a SILC client
    SILC::Client client(false);

    // Put the tensor in the database
    std::string key = "3d_tensor";
    client.put_tensor(key, input_tensor.data(), dims,
                      SILC::TensorType::dbl,
                      SILC::MemoryLayout::contiguous);

    // Retrieve the tensor from the database using the unpack feature.
    std::vector<double> unpack_tensor(n_values, 0);
    client.unpack_tensor(key, unpack_tensor.data(), {n_values},
                        SILC::TensorType::dbl,
                        SILC::MemoryLayout::contiguous);

    // Print the values retrieved with the unpack feature
    std::cout<<"Comparison of the sent and "\
                "retrieved (via unpack) values: "<<std::endl;
    for(size_t i=0; i<n_values; i++)
        std::cout<<"Sent: "<<input_tensor[i]<<" "
                 <<"Received: "<<unpack_tensor[i]<<std::endl;


    // Retrieve the tensor from the database using the get feature.
    SILC::TensorType get_type;
    std::vector<size_t> get_dims;
    void* get_tensor;
    client.get_tensor(key, get_tensor, get_dims, get_type,
                      SILC::MemoryLayout::nested);

    // Print the values retrieved with the unpack feature
    std::cout<<"Comparison of the sent and "\
                "retrieved (via get) values: "<<std::endl;
    for(size_t i=0, c=0; i<dims[0]; i++)
        for(size_t j=0; j<dims[1]; j++)
            for(size_t k=0; k<dims[2]; k++, c++) {
                std::cout<<"Sent: "<<input_tensor[c]<<" "
                         <<"Received: "
                         <<((double***)get_tensor)[i][j][k]<<std::endl;
    }

    return 0;
}
