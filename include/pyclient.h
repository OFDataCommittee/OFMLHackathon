#define PY_CLIENT_H
#ifdef __cplusplus

#include "client.h"
#include "pydataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

using namespace SILC;

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

  // Tensor Functions
  void put_tensor(std::string& key /*!< The key to use to place the tensor*/,
                  std::string& type /*!< The data type of the tensor*/,
                  py::array data /*!< Numpy array with Pybind*/
                  );
  py::array get_tensor(std::string& key);


  // Dataset Functions
  void put_dataset(PyDataset& dataset);
  PyDataset* get_dataset(const std::string& name);


  // Script functions
  //! Set a script (from file) in the database for future execution
  void set_script_from_file(const std::string& key /*!< The key to use to place the script*/,
                            const std::string& device /*!< The device to run the script*/,
                            const std::string& script_file /*!< The name of the script file*/
                            );
  //! Set a script (from buffer) in the database for future execution
  void set_script(const std::string& key /*!< The key to use to place the script*/,
                  const std::string& device /*!< The device to run the script*/,
                  const std::string_view& script /*!< The name of the script file*/
                  );
  //! Get the script from the database
  std::string_view get_script(const std::string& key /*!< The key to use to retrieve the script*/
                              );
  //! Run a script in the database
  void run_script(const std::string& key /*!< The key of the script to run*/,
                  const std::string& function /*!< The name of the function to run in the script*/,
                  std::vector<std::string>& inputs /*!< The keys of the input tensors*/,
                  std::vector<std::string>& outputs /*!< The keys of the output tensors*/
                  );

  //! Set a model (from buffer) in the database for future execution
  void set_model(const std::string& key /*!< The key to use to place the model*/,
                 const std::string_view& model /*!< The model as a continuous buffer string_view*/,
                 const std::string& backend /*!< The name of the backend (TF, TFLITE, TORCH, ONNX)*/,
                 const std::string& device /*!< The name of the device (CPU, GPU, GPU:0, GPU:1...)*/,
                 int batch_size = 0 /*!< The batch size for model execution*/,
                 int min_batch_size = 0 /*!< The minimum batch size for model execution*/,
                 const std::string& tag = "" /*!< A tag to attach to the model for information purposes*/,
                 const std::vector<std::string>& inputs
                  = std::vector<std::string>() /*!< One or more names of model input nodes (TF models)*/,
                 const std::vector<std::string>& outputs
                  = std::vector<std::string>() /*!< One or more names of model output nodes (TF models)*/
                 );

  //! Set a model (from file) in the database for future execution
  void set_model_from_file(const std::string& key /*!< The key to use to place the model*/,
                           const std::string& model_file /*!< The file storing the model*/,
                           const std::string& backend /*!< The name of the backend (TF, TFLITE, TORCH, ONNX)*/,
                           const std::string& device /*!< The name of the device (CPU, GPU, GPU:0, GPU:1...)*/,
                           int batch_size = 0 /*!< The batch size for model execution*/,
                           int min_batch_size = 0 /*!< The minimum batch size for model execution*/,
                           const std::string& tag = "" /*!< A tag to attach to the model for information purposes*/,
                           const std::vector<std::string>& inputs
                            = std::vector<std::string>() /*!< One or more names of model input nodes (TF models)*/,
                           const std::vector<std::string>& outputs
                            = std::vector<std::string>() /*!< One or more names of model output nodes (TF models)*/
                           );

  //! Run a model in the database
  void run_model(const std::string& key /*!< The key of the model to run*/,
                 std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                 std::vector<std::string> outputs /*!< The keys of the output tensors*/
                );

  //! Get a model in the database
  py::bytes get_model(const std::string& key /*!< The key to use to retrieve the model*/
                             );


private:
  Client* _client;

};
#endif //PY_CLIENT_H