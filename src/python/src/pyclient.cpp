#include "pyclient.h"

namespace py = pybind11;

SmartSimPyClient::SmartSimPyClient(bool cluster, bool fortran_array)
{
  SmartSimClient* client = new SmartSimClient(cluster, fortran_array);
  this->_client = client;
}


SmartSimPyClient::~SmartSimPyClient() {
  delete this->_client;
}

void SmartSimPyClient::put_tensor(std::string& key, std::string& type, py::array data) {

  auto buffer = data.request();
  void* ptr = buffer.ptr;

  // get dims
  std::vector<size_t> dims(buffer.ndim);
  for (int i=0; i < buffer.shape.size(); i++) {
      dims[i] = (size_t) buffer.shape[i];
  }

  this->_client->put_tensor(key, type, ptr, dims, MemoryLayout::contiguous);
  return;
}


py::array SmartSimPyClient::get_tensor(std::string& key) {

  // placeholder variables to be filled
  std::string type;
  std::vector<size_t> dims;
  void* ptr;

  // call C++ client
  this->_client->get_tensor(key, type, ptr, dims, MemoryLayout::contiguous);

  // detect data type
  int data_type = TENSOR_TYPE_MAP.at(type);
  switch(data_type) {
    case DOUBLE_TENSOR_TYPE : {
      double* data;
      data = (double*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case FLOAT_TENSOR_TYPE : {
      float* data;
      data = (float*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case INT64_TENSOR_TYPE : {
      int64_t* data;
      data = (int64_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case INT32_TENSOR_TYPE : {
      int32_t* data;
      data = (int32_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case INT16_TENSOR_TYPE : {
      int16_t* data;
      data = (int16_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case INT8_TENSOR_TYPE : {
      int8_t* data;
      data = (int8_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case UINT16_TENSOR_TYPE : {
      uint16_t* data;
      data = (uint16_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case UINT8_TENSOR_TYPE : {
      uint8_t* data;
      data = (uint8_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    default :
      // TODO throw python expection here
      throw std::runtime_error("Could not infer type");
      break;
  }
}
