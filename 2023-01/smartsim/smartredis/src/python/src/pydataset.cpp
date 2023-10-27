/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
#include "srexception.h"
#include "logger.h"

using namespace SmartRedis;

namespace py = pybind11;

// Decorator to standardize exception handling in PyBind DataSet API methods
template <class T>
auto pb_dataset_api(T&& dataset_api_func, const char* name)
{
  // we create a closure below
  auto decorated =
  [name, dataset_api_func = std::forward<T>(dataset_api_func)](auto&&... args)
  {
    try {
      return dataset_api_func(std::forward<decltype(args)>(args)...);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        std::string msg(
            "A non-standard exception was encountered while executing ");
        msg += name;
        throw SRInternalException(msg);
    }
  };
  return decorated;
}

// Macro to invoke the decorator with a lambda function
#define MAKE_DATASET_API(stuff)\
    pb_dataset_api([&] { stuff }, __func__)()


PyDataset::PyDataset(const std::string& name)
    : PySRObject(name)
{
    MAKE_DATASET_API({
        _dataset = new DataSet(name);
    });
}

PyDataset::PyDataset(DataSet* dataset)
    : PySRObject(dataset->get_context())
{
    MAKE_DATASET_API({
        _dataset = dataset;
    });
}

PyDataset::~PyDataset()
{
    MAKE_DATASET_API({
        if (_dataset != NULL) {
            delete _dataset;
            _dataset = NULL;
        }
    });
}

void PyDataset::add_tensor(const std::string& name, py::array data, std::string& type)
{
    MAKE_DATASET_API({
        auto buffer = data.request();
        void* ptr = buffer.ptr;

        // get dims
        std::vector<size_t> dims(buffer.ndim);
        for (size_t i = 0; i < buffer.shape.size(); i++) {
            dims[i] = (size_t) buffer.shape[i];
        }

        SRTensorType ttype = TENSOR_TYPE_MAP.at(type);
        _dataset->add_tensor(name, ptr, dims, ttype, SRMemLayoutContiguous);
    });
}

py::array PyDataset::get_tensor(const std::string& name)
{
    return MAKE_DATASET_API({
        TensorBase* tensor = _dataset->_get_tensorbase_obj(name);

        // Define py::capsule lambda function for destructor
        py::capsule free_when_done((void*)tensor, [](void *tensor) {
            delete reinterpret_cast<TensorBase*>(tensor);
        });

        // detect data type
        switch (tensor->type()) {
            case SRTensorTypeDouble: {
                double* data = reinterpret_cast<double*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            case SRTensorTypeFloat: {
                float* data = reinterpret_cast<float*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            case SRTensorTypeInt64: {
                int64_t* data = reinterpret_cast<int64_t*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            case SRTensorTypeInt32: {
                int32_t* data = reinterpret_cast<int32_t*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            case SRTensorTypeInt16: {
                int16_t* data = reinterpret_cast<int16_t*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            case SRTensorTypeInt8: {
                int8_t* data = reinterpret_cast<int8_t*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            case SRTensorTypeUint16: {
                uint16_t* data = reinterpret_cast<uint16_t*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            case SRTensorTypeUint8: {
                uint8_t* data = reinterpret_cast<uint8_t*>(
                    tensor->data_view(SRMemLayoutContiguous));
                return py::array(tensor->dims(), data, free_when_done);
            }
            default :
                throw SRRuntimeException("Could not infer type in "\
                                         "PyDataSet::get_tensor().");
        }
    });
}

void PyDataset::add_meta_scalar(
    const std::string& name, py::array data, std::string& type)
{
    MAKE_DATASET_API({
        auto buffer = data.request();
        void* ptr = buffer.ptr;

        SRMetaDataType ttype = METADATA_TYPE_MAP.at(type);
        _dataset->add_meta_scalar(name, ptr, ttype);
    });
}

void PyDataset::add_meta_string(const std::string& name, const std::string& data)
{
    MAKE_DATASET_API({
        _dataset->add_meta_string(name, data);
    });
}

py::array PyDataset::get_meta_scalars(const std::string& name)
{
    return MAKE_DATASET_API({
        SRMetaDataType type = SRMetadataTypeInvalid;
        size_t length = 0;
        void *ptr = NULL;

        _dataset->get_meta_scalars(name, ptr, length, type);

        // detect data type
        switch (type) {
            case SRMetadataTypeDouble: {
                double* data = reinterpret_cast<double*>(ptr);
                return py::array(length, data, py::none());
            }
            case SRMetadataTypeFloat: {
                float* data = reinterpret_cast<float*>(ptr);
                return py::array(length, data, py::none());
            }
            case SRMetadataTypeInt32: {
                int32_t* data = reinterpret_cast<int32_t*>(ptr);
                return py::array(length, data, py::none());
            }
            case SRMetadataTypeInt64: {
                int64_t* data = reinterpret_cast<int64_t*>(ptr);
                return py::array(length, data, py::none());
            }
            case SRMetadataTypeUint32: {
                uint32_t* data = reinterpret_cast<uint32_t*>(ptr);
                return py::array(length, data, py::none());
            }
            case SRMetadataTypeUint64: {
                uint64_t* data = reinterpret_cast<uint64_t*>(ptr);
                return py::array(length, data, py::none());
            }
            case SRMetadataTypeString: {
                throw SRRuntimeException("MetaData is of type string. "\
                                         "Use get_meta_strings method.");
            }
            default :
                throw SRRuntimeException("Could not infer type in "\
                                         "PyDataSet::get_meta_scalars().");
        }
    });
}

std::string PyDataset::get_name()
{
    return MAKE_DATASET_API({
        return _dataset->get_name();
    });
}

py::list PyDataset::get_meta_strings(const std::string& name)
{
    return MAKE_DATASET_API({
        // We return a copy
        return py::cast(_dataset->get_meta_strings(name));
    });
}

// Retrieve the names of all tensors in the DataSet
py::list PyDataset::get_tensor_names()
{
    return MAKE_DATASET_API({
        // We return a copy
        return py::cast(_dataset->get_tensor_names());
    });
}

// Retrieve the data type of a Tensor in the DataSet
std::string PyDataset::get_tensor_type(const std::string& name)
{
    return MAKE_DATASET_API({
        // get the type
        SRTensorType ttype = _dataset->get_tensor_type(name);
        switch (ttype) {
            case SRTensorTypeDouble:
                return "DOUBLE";
            case SRTensorTypeFloat:
                return "FLOAT";
            case SRTensorTypeInt64:
                return "INT64";
            case SRTensorTypeInt32:
                return "INT32";
            case SRTensorTypeInt16:
                return "INT16";
            case SRTensorTypeInt8:
                return "INT8";
            case SRTensorTypeUint16:
                return "UINT6";
            case SRTensorTypeUint8:
                return "UINT8";
            default :
                throw SRRuntimeException("Unrecognized type in "\
                                         "PyDataSet::get_tensor_type().");
        }
    });
}

// Retrieve the dimensions of a Tensor in the DataSet
py::list PyDataset::get_tensor_dims(const std::string& name)
{
    return MAKE_DATASET_API({
        // We return a copy
        return py::cast(_dataset->get_tensor_dims(name));
    });
}

// Retrieve the names of all metadata fields in the DataSet
py::list PyDataset::get_metadata_field_names()
{
    return MAKE_DATASET_API({
        // We return a copy
        return py::cast(_dataset->get_metadata_field_names());
    });
}

// Retrieve the data type of a metadata field in the DataSet
std::string PyDataset::get_metadata_field_type(const std::string& name)
{
    return MAKE_DATASET_API({
        // get the type
        auto mdtype = _dataset->get_metadata_field_type(name);
        for (auto it = METADATA_TYPE_MAP.begin();
             it != METADATA_TYPE_MAP.end(); ++it) {
            if (it->second == mdtype)
                return it->first;
        }
        throw SRRuntimeException("Unrecognized type in "\
                                 "PyDataSet::get_metadata_field_type().");
    });
}

// Create a string representation of the DataSet
std::string PyDataset::to_string()
{
    return MAKE_DATASET_API({
        return _dataset->to_string();
    });
}

// EOF
