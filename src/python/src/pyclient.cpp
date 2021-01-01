#include "pyclient.h"

namespace py = pybind11;

PyClient::PyClient(bool cluster, bool fortran_array)
{
  SmartSimClient* client = new SmartSimClient(cluster, fortran_array);
  this->_client = client;
}


PyClient::~PyClient() {
  delete this->_client;
}

void PyClient::put_tensor(std::string& key, std::string& type, py::array data) {

  auto buffer = data.request();
  void* ptr = buffer.ptr;

  // get dims
  std::vector<size_t> dims(buffer.ndim);
  for (int i=0; i < buffer.shape.size(); i++) {
      dims[i] = (size_t) buffer.shape[i];
  }

  TensorType ttype = TENSOR_TYPE_MAP.at(type);
  this->_client->put_tensor(key, ptr, dims, ttype, MemoryLayout::contiguous);
  return;
}


py::array PyClient::get_tensor(std::string& key) {

  // placeholder variables to be filled
  TensorType type;
  std::vector<size_t> dims;
  void* ptr;

  // call C++ client
  this->_client->get_tensor(key, ptr, dims, type, MemoryLayout::contiguous);

  // detect data type
  switch(type) {
    case TensorType::dbl : {
      double* data;
      data = (double*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case TensorType::flt : {
      float* data;
      data = (float*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case TensorType::int64 : {
      int64_t* data;
      data = (int64_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case TensorType::int32 : {
      int32_t* data;
      data = (int32_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case TensorType::int16 : {
      int16_t* data;
      data = (int16_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case TensorType::int8 : {
      int8_t* data;
      data = (int8_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case TensorType::uint16 : {
      uint16_t* data;
      data = (uint16_t*) ptr;
      return py::array(dims, data, py::none());
      break;
    }
    case TensorType::uint8 : {
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

void PyClient::put_dataset(PyDataset& dataset) {

  this->_client->put_dataset(*dataset.get());
  return;
}

PyDataset* PyClient::get_dataset(const std::string& name) {
  DataSet* data = new DataSet(this->_client->get_dataset(name));
  PyDataset* dataset = new PyDataset(*data);
  return dataset;
}

