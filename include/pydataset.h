#include "dataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

using namespace SmartRedis;

namespace py = pybind11;

class PyDataset;

class PyDataset
{
public:
  PyDataset(const std::string& name);
  PyDataset(DataSet& dataset);
  ~PyDataset();

  void add_tensor(const std::string& name,
                  py::array data,
                  std::string& type);

  py::array get_tensor(const std::string& name);

  void add_meta_scalar(const std::string& name,
                       py::array data,
                       std::string& type);

  void add_meta_string(const std::string& name,
                       const std::string& data);

  py::array get_meta_scalars(const std::string& name);

  py::list get_meta_strings(const std::string& name);


  DataSet* get();

private:
  DataSet* _dataset;
};
