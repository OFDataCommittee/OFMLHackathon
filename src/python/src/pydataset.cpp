
#include "pydataset.h"

using namespace SmartRedis;

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

py::array PyDataset::get_tensor(const std::string& name) {

  TensorType type;
  std::vector<size_t> dims;
  void* ptr;

  this->_dataset->get_tensor(name, ptr, dims, type, MemoryLayout::contiguous);

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

void PyDataset::add_meta_scalar(const std::string& name, py::array data, std::string& type) {
  
  auto buffer = data.request();
  void* ptr = buffer.ptr;

  MetaDataType ttype = METADATA_TYPE_MAP.at(type);
  this->_dataset->add_meta_scalar(name, ptr, ttype);
}

void PyDataset::add_meta_string(const std::string& name, const std::string& data) {
  
  this->_dataset->add_meta_string(name, data);
}

py::array PyDataset::get_meta_scalars(const std::string& name) {

  MetaDataType type;
  size_t length;
  void *ptr;

  this->_dataset->get_meta_scalars(name, ptr, length, type);

  // detect data type
  switch(type) {
    case MetaDataType::dbl : {
      double* data;
      data = (double*) ptr;
      return py::array(length, data, py::none());
      break;
    }
    case MetaDataType::flt : {
      float* data;
      data = (float*) ptr;
      return py::array(length, data, py::none());
      break;
    }
    case MetaDataType::int32 : {
      int32_t* data;
      data = (int32_t*) ptr;
      return py::array(length, data, py::none());
      break;
    }
    case MetaDataType::int64 : {
      int64_t* data;
      data = (int64_t*) ptr;
      return py::array(length, data, py::none());
      break;
    }
    case MetaDataType::uint32 : {
      uint32_t* data;
      data = (uint32_t*) ptr;
      return py::array(length, data, py::none());
      break;
    }
    case MetaDataType::uint64 : {
      uint64_t* data;
      data = (uint64_t*) ptr;
      return py::array(length, data, py::none());
      break;
    }
    case MetaDataType::string : {
      throw std::runtime_error("MetaData is of type string. Use get_meta_strings method.");
    }
    default :
      // TODO throw python exception here
      throw std::runtime_error("Could not infer type");
      break;
  }

}

py::list PyDataset::get_meta_strings(const std::string& name) {
  
  // We return a copy
  return py::cast(this->_dataset->get_meta_strings(name));
}
