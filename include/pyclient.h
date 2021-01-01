#define PY_CLIENT_H
#ifdef __cplusplus

#include "client.h"
#include "pydataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

class PyClient;

class PyClient
{
public:
  PyClient(
    bool cluster /*!< Flag to indicate if a database cluster is being used*/,
    bool fortran_array = false /*!< Flag to indicate if fortran arrays are being used*/
  );
  ~PyClient();

  //! Put a tensor into the database
  void put_tensor(std::string& key /*!< The key to use to place the tensor*/,
                  std::string& type /*!< The data type of the tensor*/,
                  py::array data /*!< Numpy array with Pybind*/
                  );

  py::array get_tensor(std::string& key);

  void put_dataset(PyDataset& dataset);

  PyDataset* get_dataset(const std::string& name);

private:
  SmartSimClient* _client;

};
#endif //PY_CLIENT_H