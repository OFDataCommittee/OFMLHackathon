#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

template <typename T_send, typename T_recv>
void rename_dataset(
		    void (*fill_array)(T_send***, int, int, int),
		    std::vector<size_t> dims,
            SILC::TensorType type,
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
    SILC::Client client(use_cluster());
    SILC::DataSet sent_dataset(dataset_name);

    //Add metadata to the DataSet
    DATASET_TEST_UTILS::fill_dataset_with_metadata(sent_dataset);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";
    std::string t_name_3 = "tensor_3";

    sent_dataset.add_tensor(t_name_1, t_send_1,
                        dims, type, SILC::MemoryLayout::nested);
    sent_dataset.add_tensor(t_name_2, t_send_2,
                        dims, type, SILC::MemoryLayout::nested);
    sent_dataset.add_tensor(t_name_3, t_send_3,
                        dims, type, SILC::MemoryLayout::nested);

    //Put the DataSet into the database
    client.put_dataset(sent_dataset);

    //Rename the DataSet to a new name
    std::string new_dataset_name = "renamed_" + dataset_name;
    client.rename_dataset(dataset_name, new_dataset_name);

    //Check that the old keys have been removed
    std::string key;
    key = "{"+dataset_name+"}."+t_name_1;
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet tensor " + key +
                                 "was not deleted.");

    key = "{"+dataset_name+"}."+t_name_2;
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet tensor " + key +
                                 "was not deleted.");

    key = "{"+dataset_name+"}."+t_name_3;
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet tensor " + key +
                                 "was not deleted.");

    key = "{"+dataset_name+"}.meta";
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet metadata "\
                                 "was not deleted.");

    if(client.tensor_exists(dataset_name))
        throw std::runtime_error("The DataSet confirmation "\
                                 "key was not deleted.");

    if(!client.tensor_exists(new_dataset_name))
        throw std::runtime_error("The renamed DataSet "\
                                 "confirmation key is not set.");

    SILC::DataSet retrieved_dataset = client.get_dataset(new_dataset_name);


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

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(retrieved_dataset);

    return;
}

int main(int argc, char* argv[]) {

  //Declare the dimensions for the 3D arrays
  std::vector<size_t> dims{5,4,17};

  std::string dataset_name;

  dataset_name = "3D_dbl_dataset_rank";
  rename_dataset<double,double>(
				  &set_3D_array_floating_point_values<double>,
				  dims, SILC::TensorType::dbl, "_dbl", dataset_name);

  return 0;
}
