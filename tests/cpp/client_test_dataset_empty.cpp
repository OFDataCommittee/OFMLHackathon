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
    try {
        client.put_dataset(sent_dataset);
    }
    catch(std::runtime_error) {
        return;
    }

    throw std::runtime_error("Failed to throw error "
                             "for empty DataSet.");
}

int main(int argc, char* argv[]) {

    put_get_empty_dataset("dataset_empty");
    return 0;
}
