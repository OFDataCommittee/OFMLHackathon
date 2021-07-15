/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


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

    TensorBase* tensor;
    try {
        tensor = this->_dataset->_get_tensorbase_obj(name);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }

    //Define py::capsule lambda function for destructor
    py::capsule free_when_done((void*)tensor, [](void *tensor) {
            delete (TensorBase*)tensor;
            });

    // detect data type
    switch(tensor->type()) {
        case TensorType::dbl : {
            double* data =
                (double*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::flt : {
            float* data =
                (float*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int64 : {
            int64_t* data =
                (int64_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int32: {
            int32_t* data =
                (int32_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int16: {
            int16_t* data =
                (int16_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int8: {
            int8_t* data =
                (int8_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::uint16: {
            uint16_t* data =
                (uint16_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::uint8: {
            uint8_t* data =
                (uint8_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        default :
            throw std::runtime_error("Could not infer type in "\
                                     "PyDataSet::get_tensor().");
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
