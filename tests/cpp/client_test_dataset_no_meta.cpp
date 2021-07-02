#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

template <typename T_send, typename T_recv>
void put_get_dataset(
		    void (*fill_array)(T_send***, int, int, int),
		    std::vector<size_t> dims,
            SmartRedis::TensorType type,
            std::string key_suffix,
            std::string dataset_name)
{
    T_send*** t_send_1 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_1, dims[0], dims[1], dims[2]);

    T_send*** t_send_2 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_2, dims[0], dims[1], dims[2]);

    T_send*** t_send_3 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_3, dims[0], dims[1], dims[2]);

    //Create Client and DataSet
    SmartRedis::Client client(use_cluster());
    SmartRedis::DataSet sent_dataset(dataset_name);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";
    std::string t_name_3 = "tensor_3";

    sent_dataset.add_tensor(t_name_1, t_send_1,
                        dims, type, SmartRedis::MemoryLayout::nested);
    sent_dataset.add_tensor(t_name_2, t_send_2,
                        dims, type, SmartRedis::MemoryLayout::nested);
    sent_dataset.add_tensor(t_name_3, t_send_3,
                        dims, type, SmartRedis::MemoryLayout::nested);

    //Put the DataSet into the database
    client.put_dataset(sent_dataset);

    if(!client.tensor_exists(dataset_name))
        throw std::runtime_error("The DataSet "\
                                 "confirmation key is not set.");

    SmartRedis::DataSet retrieved_dataset = client.get_dataset(dataset_name);

    DATASET_TEST_UTILS::check_tensor_names(retrieved_dataset,
                                    {t_name_1, t_name_2, t_name_3});

    //Check that the tensors are the same
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_1,
                                               type, t_send_1, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_2,
                                               type, t_send_2, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_3,
                                               type, t_send_3, dims);

    return;
}

int main(int argc, char* argv[]) {

    //Declare the dimensions for the 3D arrays
    std::vector<size_t> dims{5,4,17};

    put_get_dataset<double,double>(
                    &set_3D_array_floating_point_values<double>,
                    dims, SmartRedis::TensorType::dbl,
                    "_dbl", "dataset_no_meta");

    return 0;
}
