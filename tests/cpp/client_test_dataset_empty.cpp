#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

void put_get_empty_dataset(std::string dataset_name)
{
    //Create Client and DataSet
    SmartRedis::Client client(use_cluster());
    SmartRedis::DataSet sent_dataset(dataset_name);

    //Put the DataSet into the database
    client.put_dataset(sent_dataset);

    SmartRedis::DataSet retrieved_dataset = client.get_dataset(dataset_name);

    return;
}

int main(int argc, char* argv[]) {

    put_get_empty_dataset("dataset_empty");
    return 0;
}
