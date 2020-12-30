
#include "pydataset.h"

namespace py = pybind11;

PyDataset::PyDataset(std::string& name) {
  DataSet* dataset = new DataSet(name);
  this->_dataset = dataset;
}

PyDataset::~PyDataset() {
    delete this->_dataset;
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
