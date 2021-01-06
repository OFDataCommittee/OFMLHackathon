#include "dataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

using namespace SILC;

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

  py::array get_tensor(const std::string& key);

  DataSet* get();

private:
  DataSet* _dataset;
};