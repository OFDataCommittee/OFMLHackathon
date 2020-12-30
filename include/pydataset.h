#include "dataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

class PyDataset;

class PyDataset
{
public:
  PyDataset(std::string& name);
  ~PyDataset();

  void add_tensor(const std::string& name,
                  py::array data,
                  std::string& type);

private:
  DataSet* _dataset;

};