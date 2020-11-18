#define SMARTSIM_PY_CLIENT_H
#ifdef __cplusplus

#include "client.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

class SmartSimPyClient;

class SmartSimPyClient
{
public:
  SmartSimPyClient(
    bool cluster /*!< Flag to indicate if a database cluster is being used*/,
    bool fortran_array = false /*!< Flag to indicate if fortran arrays are being used*/
  );
  ~SmartSimPyClient();

  //! Put a tensor into the database
  void put_tensor(std::string& key /*!< The key to use to place the tensor*/,
                  std::string& type /*!< The data type of the tensor*/,
                  py::array data /*!< Numpy array with Pybind*/
                  );
private:
  SmartSimClient* _client;

};
#endif //SMARTSIM_PY_CLIENT_H