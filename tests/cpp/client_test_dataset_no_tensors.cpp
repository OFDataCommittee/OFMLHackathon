#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

void put_dataset_no_tensors(std::string dataset_name)
{
    //Create Client and DataSet
    SmartRedis::Client client(use_cluster());
    SmartRedis::DataSet sent_dataset(dataset_name);

    //Add metadata to the DataSet
    DATASET_TEST_UTILS::fill_dataset_with_metadata(sent_dataset);

    //Put the DataSet into the database
    client.put_dataset(sent_dataset);

    if(!client.tensor_exists(dataset_name))
        throw std::runtime_error("The DataSet "\
                                 "confirmation key is not set.");

    SmartRedis::DataSet retrieved_dataset = client.get_dataset(dataset_name);

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(retrieved_dataset);

    return;
}

int main(int argc, char* argv[]) {

    put_dataset_no_tensors("dataset_no_tensors");
    return 0;
}
