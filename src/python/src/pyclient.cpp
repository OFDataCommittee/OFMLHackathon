#include "pyclient.h"

namespace py = pybind11;

SmartSimPyClient::SmartSimPyClient(bool cluster, bool fortran_array)
{
    SmartSimClient* client = new SmartSimClient(cluster, fortran_array);
    this->_client = client;
}
// put tensor

SmartSimPyClient::~SmartSimPyClient() {
  delete this->_client;
}

void SmartSimPyClient::put_tensor(std::string& key, std::string& type, py::array data) {

    auto buffer = data.request();
    void* ptr = buffer.ptr;

    // get dims
    std::vector<int> dims(buffer.ndim);
    for (int i=0; i < buffer.shape.size(); i++) {
        dims[i] = (int) buffer.shape[i];
    }

    this->_client->put_tensor(key, type, ptr, dims);
    return;
    }
