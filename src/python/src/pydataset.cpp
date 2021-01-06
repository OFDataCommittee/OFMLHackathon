
#include "pydataset.h"

using namespace SILC;

namespace py = pybind11;

PyDataset::PyDataset(const std::string& name) {
  DataSet* dataset = new DataSet(name);
  this->_dataset = dataset;
}

PyDataset::PyDataset(DataSet& dataset) {
  this->_dataset = &dataset;
}

PyDataset::~PyDataset() {
  delete this->_dataset;
}

DataSet* PyDataset::get() {
  return this->_dataset;
}

void PyDataset::add_tensor(const std::string& name, py::array data, std::string& type) {

  auto buffer = data.request();
  void* ptr = buffer.ptr;

  // get dims
  std::vector<size_t> dims(buffer.ndim);
  for (int i=0; i < buffer.shape.size(); i++) {
      dims[i] = (size_t) buffer.shape[i];
  }

  TensorType ttype = TENSOR_TYPE_MAP.at(type);
  this->_dataset->add_tensor(name, ptr, dims, ttype, MemoryLayout::contiguous);
  return;
}

py::array PyDataset::get_tensor(const std::string& key) {

  TensorType type;
  std::vector<size_t> dims;
  void* ptr;

  this->_dataset->get_tensor(key, ptr, dims, type, MemoryLayout::contiguous);

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