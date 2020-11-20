#ifndef SMARTSIM_C_CLIENT_H
#define SMARTSIM_C_CLIENT_H
///@file
///\brief C-wrappers for the C++ SmartSimClient class
#include <stdlib.h>
#include <stdbool.h>
#include "client.h"
#ifdef __cplusplus
extern "C" {
#endif

//! SmartSimClient C-client constructor
void* SmartSimCClient(bool cluster /*!< Flag to indicate if a database cluster is being used*/
                     );

//! SmartSimClient C-client destructor
void DeleteCClient(void* c_client /*!< The c client to use for communication*/
                   );

//! Put a DataSet object into the database
void put_dataset(void* c_client /*!< The c client to use for communication*/,
                 const void* dataset /*!< The DataSet object to send*/
                 );

//! Get a DataSet object from the database
void* get_dataset(void* c_client /*!< The c client to use for communication*/,
                  const char* name /*!< The name of the dataset object to fetch*/,
                  const size_t name_length /*!< The length of the name c-string, excluding null terminating character */
                  );

//! Move a DataSet to a new key
void rename_dataset(void* c_client /*!< The c client to use for communication*/,
                    const char* name /*!< The name of the dataset object*/,
                    const size_t name_length /*!< The length of the name c-string, excluding null terminating character */,
                    const char* new_name /*!< The name of the dataset object*/,
                    const size_t new_name_length /*!< The length of the new name c-string, excluding null terminating character */
                    );

//! Copy a DataSet to a new key
void copy_dataset(void* c_client /*!< The c client to use for communication*/,
                  const char* src_name /*!< The source name of the dataset object*/,
                  const size_t src_name_length /*!< The length of the src_name c-string, excluding null terminating character */,
                  const char* dest_name /*!< The destination name of the dataset object*/,
                  const size_t dest_name_length /*!< The length of the dest_name c-string, excluding null terminating character */
                  );

//! Delete a DataSet
void delete_dataset(void* c_client /*!< The c client to use for communication*/,
                    const char* name /*!< The name of the dataset object*/,
                    const size_t name_length /*!< The length of the name c-string, excluding null terminating character */
                    );

//! Put a tensor into the database
void put_tensor(void* c_client /*!< The c client to use for communication*/,
                const char* key /*!< The key to use to place the tensor*/,
                const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                const char* type /*!< The data type of the tensor*/,
                const size_t type_length /*!< The length of the type c-string, excluding null terminating character */,
                void* data /*!< A c ptr to the beginning of the data*/,
                const size_t* dims /*!< Length along each dimension of the tensor*/,
                const size_t n_dims /*!< The number of dimensions of the tensor*/
                );

//! Get a tensor from the database and fill the provided memory space (result) that is layed out as defined by dims
void get_tensor(void* c_client /*!< The c client to use for communication*/,
                const char* key /*!< The key to use to fetch the tensor*/,
                const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                char** type /*!< The data type of the tensor*/,
                size_t* type_length /*!< The length of the type c-string, excluding null terminating character */,
                void** data /*!< A c ptr to the beginning of the result array to fill*/,
                size_t** dims /*!< The dimensions of the tensor*/,
                size_t* n_dims /*!< The number of dimensions of the tensor*/
                );

//! Get a tensor from the database and fill the provided memory space (result) that is layed out as defined by dims
void unpack_tensor(void* c_client /*!< The c client to use for communication*/,
                  const char* key /*!< The key to use to fetch the tensor*/,
                  const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                  const char* type /*!< The data type of the tensor*/,
                  const size_t type_length /*!< The length of the type c-string, excluding null terminating character */,
                  void* result /*!< A c ptr to the beginning of the result array to fill*/,
                  const size_t* dims /*!< The dimensions of the tensor*/,
                  const size_t n_dims /*!< The number of dimensions of the tensor*/
                  );

//! Move a tensor to a new key
void rename_tensor(void* c_client /*!< The c client to use for communication*/,
                   const char* key /*!< The key to use to fetch the tensor*/,
                   const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                   const char* new_key /*!< The new tensor key*/,
                   const size_t new_key_length /*!< The length of the new_key c-string, excluding null terminating character */
                   );

//! Delete a tensor
void delete_tensor(void* c_client /*!< The c client to use for communication*/,
                   const char* key /*!< The key to use to fetch the tensor*/,
                   const size_t key_length /*!< The length of the key c-string, excluding null terminating character */
                   );

//! This method will copy a tensor to the destination key
void copy_tensor(void* c_client /*!< The c client to use for communication*/,
                 const char* src_name /*!< The source name of the tensor*/,
                 const size_t src_name_length /*!< The length of the src_name c-string, excluding null terminating character */,
                 const char* dest_name /*!< The destination name of the tensor*/,
                 const size_t dest_name_length /*!< The length of the dest_name c-string, excluding null terminating character */
                 );

//! Set a model (from file) in the database for future execution
void set_model_from_file(void* c_client /*!< The c client to use for communication*/,
                         const char* key /*!< The key to use to place the model*/,
                         const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                         const char* model_file /*!< The file storing the model*/,
                         const size_t model_file_length /*!< The length of the file name c-string, excluding null terminating character*/,
                         const char* backend /*!< The name of the backend (TF, TFLITE, TORCH, ONNX)*/,
                         const size_t backend_length /*!< The length of the backend name c-string, excluding null terminating character*/,
                         const char* device /*!< The name of the device (CPU, GPU, GPU:0, GPU:1...)*/,
                         const size_t device_length  /*!< The length of the device name c-string, excluding null terminating character*/,
                         const int batch_size /*!< The batch size for model execution*/,
                         const int min_batch_size /*!< The minimum batch size for model execution*/,
                         const char* tag /*!< A tag to attach to the model for information purposes*/,
                         const size_t tag_length /*!< The length of the tag c-string, excluding null terminating character */,
                         const char** inputs /*!< One or more names of model input nodes (TF models) */,
                         const int* input_lengths /*!< The length of each input name c-string, excluding null terminating character*/,
                         const int n_inputs /*!< The number of inputs*/,
                         const char** outputs /*!< One or more names of model output nodes (TF models) */,
                         const int* output_lengths /*!< The length of each output name c-string, excluding null terminating character*/,
                         const int n_outputs /*!< The number of outputs*/
                         );

//! Set a model (from buffer) in the database for future execution
void set_model(void* c_client /*!< The c client to use for communication*/,
               const char* key /*!< The key to use to place the model*/,
               const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
               const char* model /*!< The model as a continuous buffer*/,
               const size_t model_length /*!< The length of the model buffer c-string, excluding null terminating character*/,
               const char* backend /*!< The name of the backend (TF, TFLITE, TORCH, ONNX)*/,
               const size_t backend_length /*!< The length of the backend name c-string, excluding null terminating character*/,
               const char* device /*!< The name of the device (CPU, GPU, GPU:0, GPU:1...)*/,
               const size_t device_length  /*!< The length of the device name c-string, excluding null terminating character*/,
               const int batch_size /*!< The batch size for model execution*/,
               const int min_batch_size /*!< The minimum batch size for model execution*/,
               const char* tag /*!< A tag to attach to the model for information purposes*/,
               const size_t tag_length /*!< The length of the tag c-string, excluding null terminating character */,
               const char** inputs /*!< One or more names of model input nodes (TF models) */,
               const int* input_lengths /*!< The length of each input name c-string, excluding null terminating character*/,
               const int n_inputs /*!< The number of inputs*/,
               const char** outputs /*!< One or more names of model output nodes (TF models) */,
               const int* output_lengths /*!< The length of each output name c-string, excluding null terminating character*/,
               const int n_outputs /*!< The number of outputs*/
               );

//! Get a model in the database
void get_model(void* c_client /*!< The c c lient to use for communication*/,
               const char* key /*!< The key to use to get the model*/,
               const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
               const char** model /*!< The model as a continuous buffer*/,
               size_t* model_length /*!< The length of the model buffer c-string, excluding null terminating character*/
               );

//! Set a script (from file) in the database for future execution
void set_script_from_file(void* c_client /*!< The c client to use for communication*/,
                          const char* key /*!< The key to use to place the script*/,
                          const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                          const char* device /*!< The name of the device (CPU, GPU, GPU:0, GPU:1...)*/,
                          const size_t device_length  /*!< The length of the device name c-string, excluding null terminating character*/,
                          const char* script_file /*!< The name of the script file*/,
                          const size_t script_file_length  /*!< The length of the script file name c-string, excluding null terminating character*/
                          );

//! Set a script (from buffer) in the database for future execution
void set_script(void* c_client /*!< The c client to use for communication*/,
                const char* key /*!< The key to use to place the script*/,
                const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                const char* device /*!< The name of the device (CPU, GPU, GPU:0, GPU:1...)*/,
                const size_t device_length  /*!< The length of the device name c-string, excluding null terminating character*/,
                const char* script /*!< The script as a c-string buffer*/,
                const size_t script_length  /*!< The length of the script c-string, excluding null terminating character*/
                );

//! Get the script from the database
void get_script(void* c_client /*!< The c client to use for communication*/,
                const char* key /*!< The key to use to get the model*/,
                const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                const char** script /*!< The script as a continuous buffer*/,
                size_t* script_length /*!< The length of the script buffer c-string, excluding null terminating character*/
                );

//! Run a script in the database
void run_script(void* c_client /*!< The c client to use for communication*/,
                const char* key /*!< The key to use to place the script*/,
                const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
                const char* function /*!< The name of the function in the script to call*/,
                const size_t function_length /*!< The length of the function name c-string, excluding null terminating character */,
                const char** inputs /*!< One or more names of model input nodes (TF models) */,
                const int* input_lengths /*!< The length of each input name c-string, excluding null terminating character*/,
                const int n_inputs /*!< The number of inputs*/,
                const char** outputs /*!< One or more names of model output nodes (TF models) */,
                const int* output_lengths /*!< The length of each output name c-string, excluding null terminating character*/,
                const int n_outputs /*!< The number of outputs*/
                );

//! Run a model in the database
void run_model(void* c_client /*!< The c client to use for communication*/,
               const char* key /*!< The key to use to run the model*/,
               const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
               const char** inputs /*!< One or more names of model input nodes (TF models) */,
               const int* input_lengths /*!< The length of each input name c-string, excluding null terminating character*/,
               const int n_inputs /*!< The number of inputs*/,
               const char** outputs /*!< One or more names of model output nodes (TF models) */,
               const int* output_lengths /*!< The length of each output name c-string, excluding null terminating character*/,
               const int n_outputs /*!< The number of outputs*/
               );

//! Check if a key exists
bool key_exists(void* c_client /*!< The c client to use for communication*/,
                const char* key /*!< The key to check*/,
                const size_t key_length /*!< The length of the key c-string, excluding null terminating character */
                );

//! Poll database until key exists or number of tries is exceeded
bool poll_key(void* c_client /*!< The c client to use for communication*/,
              const char* key /*!< The key to check*/,
              const size_t key_length /*!< The length of the key c-string, excluding null terminating character */,
              const int poll_frequency_ms /*!< The frequency of polls*/,
              const int num_tries /*!< The total number of tries*/
              );

#ifdef __cplusplus
}
#endif
#endif // SMARTSIM_C_CLIENT_H
