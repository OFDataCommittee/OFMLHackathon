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

#ifndef SMARTREDIS_C_CLIENT_H
#define SMARTREDIS_C_CLIENT_H

#include <stdlib.h>
#include <stdbool.h>
#include "client.h"
#include "sr_enums.h"
#include "srexception.h"

///@file
///\brief C-wrappers for the C++ Client class

#ifdef __cplusplus
extern "C" {
#endif

/*!
*   \brief C-client simple constructor that uses default environment variables
*          to locate configuration settings
*   \param logger_name Identifier for the current client
*   \param logger_name_length Length in characters of the logger_name string
*   \param new_client Receives the new client
*   \return Returns SRNoError on success or an error code on failure
*/
SRError SimpleCreateClient(
    const char* logger_name,
    const size_t logger_name_length,
    void** new_client);

/*!
*   \brief C-client constructor that uses a ConfigOptions object
*          to locate configuration settings
*   \param config_options The ConfigOptions object to use
*   \param logger_name Identifier for the current client
*   \param logger_name_length Length in characters of the logger_name string
*   \param new_client Receives the new client
*   \return Returns SRNoError on success or an error code on failure
*/
SRError CreateClient(
    void* config_options,
    const char* logger_name,
    const size_t logger_name_length,
    void** new_client);

/*!
*   \brief C-client constructor
*   \param cluster Flag to indicate if a database cluster is being used
*   \param logger_name Identifier for the current client
*   \param logger_name_length Length in characters of the logger_name string
*   \param new_client Receives the new client
*   \return Returns SRNoError on success or an error code on failure
*/
SRError SmartRedisCClient(
    bool cluster,
    const char* logger_name,
    const size_t logger_name_length,
    void **new_client);



/*!
*   \brief C-client constructor (deprecated)
*   \param cluster Flag to indicate if a database cluster is being used
*   \param logger_name Identifier for the current client
*   \param logger_name_length Length in characters of the logger_name string
*   \param new_client Receives the new client
*   \return Returns SRNoError on success or an error code on failure
*/
SRError SmartRedisCClient(
    bool cluster,
    const char* logger_name,
    const size_t logger_name_length,
    void **new_client);

/*!
*   \brief C-client destructor
*   \param c_client A pointer to a pointer to the c-client to destroy.
*                   The client is set to NULL on completion
*   \return Returns SRNoError on success or an error code on failure
*/
SRError DeleteCClient(void** c_client);

/*!
*   \brief Put a DataSet object into the database
*   \details The final dataset key under which the dataset is stored
*            is generated from the name that was supplied when the
*            dataset was created and may be prefixed. See
*            use_dataset_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param dataset The DataSet object to send
*   \return Returns SRNoError on success or an error code on failure
*/
SRError put_dataset(void* c_client, void* dataset);

/*!
*   \brief Get a DataSet object from the database
*   \details The final dataset key used to locate the dataset
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source() and
*            use_dataset_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the dataset object to fetch
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param dataset Receives the DataSet
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_dataset(void* c_client,
                    const char* name,
                    const size_t name_length,
                    void** dataset);

/*!
*   \brief Move a DataSet to a new name
*   \details The old and new dataset keys used to
*            find and relocate the dataset may be formed by applying
*            prefixes to the supplied old_name and new_name.
*            See set_data_source() and use_dataset_ensemble_prefix()
*            for more details.
*   \param c_client The client object to use for communication
*   \param old_name The current name key of the dataset object
*   \param old_name_length The length of the current name string,
*                          excluding null terminating character
*   \param new_name The new name key for the dataset object
*   \param new_name_length The length of the new name string,
*                          excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError rename_dataset(void* c_client,
                       const char* old_name,
                       const size_t old_name_length,
                       const char* new_name,
                       const size_t new_name_length);

/*!
*   \brief Copy a DataSet to a new name
*   \details The source and destination dataset keys used to
*            locate and store the dataset may be formed by applying
*            prefixes to the supplied src_name and dest_name.
*            See set_data_source() and use_dataset_ensemble_prefix()
*            for more details.
*   \param c_client The client object to use for communication
*   \param src_name The source name of the dataset object
*   \param src_name_length The length of the src_name string,
*                          excluding null terminating character
*   \param dest_name The destination name for the dataset object
*   \param dest_name_length The length of the dest_name string,
*                           excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError copy_dataset(void* c_client,
                     const char* src_name,
                     const size_t src_name_length,
                     const char* dest_name,
                     const size_t dest_name_length);

/*!
*   \brief Delete a DataSet
*   \details The dataset key used to locate the dataset to be deleted
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_dataset_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the dataset object
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError delete_dataset(void* c_client,
                       const char* name,
                       const size_t name_length);

/*!
*   \brief Put a tensor into the database
*   \details The key under which the tensor is stored
*            may be formed by applying a prefix to the supplied
*            name. See use_tensor_ensemble_prefix()
*            for more details.
*   \param c_client The client object to use for communication
*   \param name The name by which the tensor should be accessed
*   \param name_length The length of the tensor name string,
*                      excluding null terminating character
*   \param data The data to store with the tensor
*   \param dims The number of elements for each dimension of the tensor
*   \param n_dims The number of dimensions of the tensor
*   \param type The data type of the tensor
*   \param mem_layout The memory layout of the data
*   \return Returns SRNoError on success or an error code on failure
*/
SRError put_tensor(void* c_client,
                   const char* name,
                   const size_t name_length,
                   void* data,
                   const size_t* dims,
                   const size_t n_dims,
                   SRTensorType type,
                   SRMemoryLayout mem_layout);

/*!
*   \brief Get the data, dimensions, and type for a tensor in the
*          database. This function will allocate and retain management of the
*          memory for the tensor data. The number of dimensions and
*          the tensor type will be set based on the tensor retrieved from the
*          database.  The requested memory layout will be used to shape the
*          returned memory space pointed to by the data pointer.
*   \details The final tensor key used to retrieve the tensor
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_tensor_ensemble_prefix() for more details.
*
*            The memory returned in data is valid until the client is
*            destroyed. This method is meant to be used when the dimensions
*            and type of the tensor are unknown or the user does not want to
*            manage memory.  However, given that the memory associated with the
*            return data is valid until client destruction, this method should
*            not be used repeatedly for large tensor data.  Instead  it is
*            recommended that the user use unpack_tensor() for large tensor
*            data and to limit memory use by the client.
*   \param c_client The client object to use for communication
*   \param name The name by which the tensor should be accessed
*   \param name_length The length of the supplied name string,
*                      excluding null terminating character
*   \param data Receives tensor data in newly allocated memory
*   \param dims Receives the number of elements in each dimension of the
*               tensor in newly allocated memory
*   \param n_dims Receives the number of dimensions for the tensor
*   \param type Receives the data type for the tensor as retrieved from
*               the database
*   \param mem_layout The layout requested for the allocated memory space
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_tensor(void* c_client,
                   const char* name,
                   const size_t name_length,
                   void** data,
                   size_t** dims,
                   size_t* n_dims,
                   SRTensorType* type,
                   SRMemoryLayout mem_layout);
/*!
*   \brief Retrieve a tensor from the database into memory provided
*          by the caller
*   \details The final tensor key used to retrieve the tensor
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_tensor_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name by which the tensor should be accessed
*   \param name_length The length of the supplied name string,
*                      excluding null terminating character
*   \param result The data buffer into which the tensor data should
*                 be written
*   \param dims The number of elements in each dimension of the
*               provided memory space
*   \param n_dims The number of dimensions in the provided memory space
*   \param type The data type for the provided memory space.
*   \param mem_layout The memory layout for the provided memory space.
*   \return Returns SRNoError on success or an error code on failure
*/
SRError unpack_tensor(void* c_client,
                     const char* name,
                     const size_t name_length,
                     void* result,
                     const size_t* dims,
                     const size_t n_dims,
                     SRTensorType type,
                     SRMemoryLayout mem_layout);

/*!
*   \brief Move a tensor to a new name
*   \details The old and new tensor keys used to find and
*            relocate the tensor may be formed by applying
*            prefixes to the supplied old_name and new_name.
*            See set_data_source()
*            and use_tensor_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param old_name The original name by which the tensor should be accessed
*   \param old_name_length The length of the old_name string,
*                          excluding null terminating character
*   \param new_name The new tensor name
*   \param new_name_length The length of the supplied new_name string,
                           excluding null terminating characters
*   \return Returns SRNoError on success or an error code on failure
*/
SRError rename_tensor(void* c_client,
                      const char* old_name,
                      const size_t old_name_length,
                      const char* new_name,
                      const size_t new_name_length);

/*!
*   \brief Delete a tensor from the database
*   \details The final tensor key used to find the tensor to be
*            deleted may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_tensor_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The tensor name for the tensor to delete
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError delete_tensor(void* c_client,
                      const char* name,
                      const size_t name_length);

/*!
*   \brief Copy a tensor to a destination tensor name
*   \details The source and destination tensor keys used to locate
*            and store the tensor may be formed by applying prefixes
*            to the supplied src_name and dest_name.
*            See set_data_source()
*            and use_tensor_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param src_name The source name from which the tensor should be copied
*   \param src_name_length The length of the src_name string,
*                          excluding null terminating character
*   \param dest_name The destination name to which the tensor should be copied
*   \param dest_name_length The length of the dest_name string,
*                           excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError copy_tensor(void* c_client,
                    const char* src_name,
                    const size_t src_name_length,
                    const char* dest_name,
                    const size_t dest_name_length);

/*!
*   \brief Determine whether the backend is TensorFlow or TensorFlowLite
*   \details Within the C SmartRedis Client, additional translation needs
             to be done to the input and output vectors for calls to
             run_model. This function checks that the backend doesn't match
             TF or TFLITE
*   \param backend The name of the backend (TF, TFLITE, TORCH, ONNX)
*/
bool _isTensorFlow(const char* backend);

/*!
*   \brief Check parameters for all parameters common to set_model methods
*   \details Make sure that all pointers are not void and that the size
*            of the inputs and outputs is not zero
*   \param c_client The client object to use for communication
*   \param name The name to associate with the model
*   \param backend The name of the backend (TF, TFLITE, TORCH, ONNX)
*   \param inputs One or more names of model input nodes (TF models only)
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes (TF models only)
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*/
void _check_params_set_model(void* c_client,
                            const char* name,
                            const char* backend,
                            const char** inputs,
                            const size_t* input_lengths,
                            const size_t n_inputs,
                            const char** outputs,
                            const size_t* output_lengths,
                            const size_t n_outputs);

/*!
*   \brief Set a model (from file) in the database for future execution
*   \details The final model key used to store the model
*            may be formed by applying a prefix to the supplied
*            name. Similarly, the tensor names in the
*            input and output nodes for TF models may be prefixed.
*            See set_data_source(), use_model_ensemble_prefix(), and
*            use_tensor_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the model
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param model_file The source file for the model
*   \param model_file_length The length of the model_file string,
*                            excluding null terminating character
*   \param backend The name of the backend (TF, TFLITE, TORCH, ONNX)
*   \param backend_length The length of the backend string,
*                        excluding null terminating character
*   \param device The name of the device for execution. May be either
*                 CPU or GPU. If multiple GPUs are present, a specific
*                 GPU can be targeted by appending its zero-based
*                 index, i.e. "GPU:1"
*   \param device_length The length of the device string,
*                        excluding null terminating character
*   \param batch_size The batch size for model execution
*   \param min_batch_size The minimum batch size for model execution
*   \param min_batch_timeout Max time (ms) to wait for min batch size
*   \param tag A tag to attach to the model for information purposes
*   \param tag_length The length of the tag string,
*                     excluding null terminating character
*   \param inputs One or more names of model input nodes (TF models only)
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes (TF models only)
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_model_from_file(void* c_client,
                            const char* name,
                            const size_t name_length,
                            const char* model_file,
                            const size_t model_file_length,
                            const char* backend,
                            const size_t backend_length,
                            const char* device,
                            const size_t device_length,
                            const int batch_size,
                            const int min_batch_size,
                            const int min_batch_timeout,
                            const char* tag,
                            const size_t tag_length,
                            const char** inputs,
                            const size_t* input_lengths,
                            const size_t n_inputs,
                            const char** outputs,
                            const size_t* output_lengths,
                            const size_t n_outputs);

/*!
*   \brief Set a model from file in the database for future
*          execution in a multi-GPU system
*   \details The final model key used to store the model
*            may be formed by applying a prefix to the supplied
*            name. Similarly, the tensor names in the
*            input and output nodes for TF models may be prefixed.
*            See set_data_source(), use_model_ensemble_prefix(), and
*            use_tensor_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the model
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param model_file The source file for the model
*   \param model_file_length The length of the model_file string,
*                            excluding null terminating character
*   \param backend The name of the backend (TF, TFLITE, TORCH, ONNX)
*   \param backend_length The length of the backend string,
*                        excluding null terminating character
*   \param first_gpu the first gpu (zero-based) to use with the model
*   \param num_gpus the number of gpus to use with the model
*   \param batch_size The batch size for model execution
*   \param min_batch_size The minimum batch size for model execution
*   \param min_batch_timeout Max time (ms) to wait for min batch size
*   \param tag A tag to attach to the model for information purposes
*   \param tag_length The length of the tag string,
*                     excluding null terminating character
*   \param inputs One or more names of model input nodes (TF models only)
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes (TF models only)
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_model_from_file_multigpu(void* c_client,
                                     const char* name,
                                     const size_t name_length,
                                     const char* model_file,
                                     const size_t model_file_length,
                                     const char* backend,
                                     const size_t backend_length,
                                     const int first_gpu,
                                     const int num_gpus,
                                     const int batch_size,
                                     const int min_batch_size,
                                     const int min_batch_timeout,
                                     const char* tag,
                                     const size_t tag_length,
                                     const char** inputs,
                                     const size_t* input_lengths,
                                     const size_t n_inputs,
                                     const char** outputs,
                                     const size_t* output_lengths,
                                     const size_t n_outputs);
/*!
*   \brief Set a model (from buffer) in the database for future execution
*   \details The final model key used to store the model
*            may be formed by applying a prefix to the supplied
*            name. Similarly, the tensor names in the
*            input and output nodes for TF models may be prefixed.
*            See set_data_source(), use_model_ensemble_prefix(), and
*            use_tensor_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the model
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param model The model as a continuous buffer
*   \param model_length The length of the model string,
*                       excluding null terminating character
*   \param backend The name of the backend (TF, TFLITE, TORCH, ONNX)
*   \param backend_length The length of the backend string,
*                        excluding null terminating character
*   \param device The name of the device for execution. May be either
*                 CPU or GPU. If multiple GPUs are present, a specific
*                 GPU can be targeted by appending its zero-based
*                 index, i.e. "GPU:1"
*   \param device_length The length of the device string,
*                        excluding null terminating character
*   \param batch_size The batch size for model execution
*   \param min_batch_size The minimum batch size for model execution
*   \param min_batch_timeout Max time (ms) to wait for min batch size
*   \param tag A tag to attach to the model for information purposes
*   \param tag_length The length of the tag string,
*                     excluding null terminating character
*   \param inputs One or more names of model input nodes (TF models only)
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes (TF models only)
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_model(void* c_client,
                  const char* name,
                  const size_t name_length,
                  const char* model,
                  const size_t model_length,
                  const char* backend,
                  const size_t backend_length,
                  const char* device,
                  const size_t device_length,
                  const int batch_size,
                  const int min_batch_size,
                  const int min_batch_timeout,
                  const char* tag,
                  const size_t tag_length,
                  const char** inputs,
                  const size_t* input_lengths,
                  const size_t n_inputs,
                  const char** outputs,
                  const size_t* output_lengths,
                  const size_t n_outputs);

 /*!
*   \brief Set a model (from buffer) in the database for future execution
*          in a multi-GPU system
*   \details The final model key used to store the model
*            may be formed by applying a prefix to the supplied
*            name. Similarly, the tensor names in the
*            input and output nodes for TF models may be prefixed.
*            See set_data_source(), use_model_ensemble_prefix(), and
*            use_tensor_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the model
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param model The model as a continuous buffer
*   \param model_length The length of the model string,
*                       excluding null terminating character
*   \param backend The name of the backend (TF, TFLITE, TORCH, ONNX)
*   \param backend_length The length of the backend string,
*                        excluding null terminating character
*   \param first_gpu the first GPU (zero-based) to use with the model
*   \param num_gpus The number of GPUs to use with the model
*   \param batch_size The batch size for model execution
*   \param min_batch_size The minimum batch size for model execution
*   \param min_batch_timeout Max time (ms) to wait for min batch size
*   \param tag A tag to attach to the model for information purposes
*   \param tag_length The length of the tag string,
*                     excluding null terminating character
*   \param inputs One or more names of model input nodes (TF models only)
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes (TF models only)
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_model_multigpu(void* c_client,
                  const char* name,
                  const size_t name_length,
                  const char* model,
                  const size_t model_length,
                  const char* backend,
                  const size_t backend_length,
                  const int first_gpu,
                  const int num_gpus,
                  const int batch_size,
                  const int min_batch_size,
                  const int min_batch_timeout,
                  const char* tag,
                  const size_t tag_length,
                  const char** inputs,
                  const size_t* input_lengths,
                  const size_t n_inputs,
                  const char** outputs,
                  const size_t* output_lengths,
                  const size_t n_outputs);

/*!
*   \brief Get a model in the database. The memory associated with the
*          model string is valid until the client is destroyed.
*   \details The model key used to locate the model
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_model_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name to use to get the model
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param model_length The length of the model buffer string,
*                       excluding null terminating character
*   \param model Receives the model as a string
*   \returns Returns SRNoError on success or an error code on failure
*/
SRError get_model(void* c_client,
                  const char* name,
                  const size_t name_length,
                  size_t* model_length,
                  const char** model);

/*!
*   \brief Set a script from file in the database for future execution
*   \details The final script key used to store the script
*            may be formed by applying a prefix to the supplied
*            name. See use_model_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param device The name of the device for execution. May be either
*                 CPU or GPU. If multiple GPUs are present, a specific
*                 GPU can be targeted by appending its zero-based
*                 index, i.e. "GPU:1"
*   \param device_length The length of the device name string,
*                        excluding null terminating character
*   \param script_file The source file for the script
*   \param script_file_length The length of the script file name string,
*                             excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_script_from_file(void* c_client,
                             const char* name,
                             const size_t name_length,
                             const char* device,
                             const size_t device_length,
                             const char* script_file,
                             const size_t script_file_length);

/*!
*   \brief Set a script from file in the database for future execution
*          in a multi-GPU system
*   \details The final script key used to store the script
*            may be formed by applying a prefix to the supplied
*            name. See use_model_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param script_file The source file for the script
*   \param script_file_length The length of the script file name string,
*                             excluding null terminating character
*   \param first_gpu the first gpu (zero-based) to use with the model
*   \param num_gpus the number of gpus to use with the model
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_script_from_file_multigpu(void* c_client,
                                      const char* name,
                                      const size_t name_length,
                                      const char* script_file,
                                      const size_t script_file_length,
                                      const int first_gpu,
                                      const int num_gpus);

/*!
*   \brief Set a script (from buffer) in the database for future execution
*          in a multi-GPU system
*   \details The final script key used to store the script
*            may be formed by applying a prefix to the supplied
*            name. See use_model_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param device The name of the device for execution. May be either
*                 CPU or GPU. If multiple GPUs are present, a specific
*                 GPU can be targeted by appending its zero-based
*                 index, i.e. "GPU:1"
*   \param device_length The length of the device name string,
*                        excluding null terminating character
*   \param script The script as a string buffer
*   \param script_length The length of the script string,
*                        excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_script(void* c_client,
                   const char* name,
                   const size_t name_length,
                   const char* device,
                   const size_t device_length,
                   const char* script,
                   const size_t script_length);

/*!
*   \brief Set a script (from buffer) in the database for future execution
*          execution in a multi-GPU system
*   \details The final script key used to store the script
*            may be formed by applying a prefix to the supplied
*            name. See use_model_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name to associate with the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param first_gpu the first gpu (zero-based) to use with the model
*   \param num_gpus the number of gpus to use with the model
*   \param script The script as a string buffer
*   \param script_length The length of the script string,
*                        excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_script_multigpu(void* c_client,
                   const char* name,
                   const size_t name_length,
                   const char* script,
                   const size_t script_length,
                   const int first_gpu,
                   const int num_gpus);

/*!
*   \brief Get a script from the database.  The memory associated with the
*          script string is valid until the client is destroyed.
*   \details The script key used to locate the script
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_model_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name to use to get the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param script Receives the script in an allocated memory buffer
*   \param script_length The length of the script buffer string,
*                        excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_script(void* c_client,
                   const char* name,
                   const size_t name_length,
                   const char** script,
                   size_t* script_length);

/*!
*   \brief Check parameters for all parameters common to set_model methods
*   \details Make sure that all pointers are not void and that the size
*            of the inputs and outputs is not zero 
*   \param c_client The client object to use for communication
*   \param name The name associated with the script
*   \param function The name of the function in the script to run
*   \param inputs The tensor keys of inputs tensors to use in the script
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs The tensor keys of output tensors that will be used
*                  to save script results
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*/
void _check_params_run_script(void* c_client,
                              const char* name,
                              const char* function,
                              const char** inputs,
                              const size_t* input_lengths,
                              const size_t n_inputs,
                              const char** outputs,
                              const size_t* output_lengths,
                              const size_t n_outputs);

/*!
*   \brief Run a script function in the database using the specificed input
*          and output tensors
*   \details The script key used to locate the script to be run
*            may be formed by applying a prefix to the supplied
*            name. Similarly, the tensor names in the
*            input and output arrays may be prefixed.
*            See set_data_source(), use_model_ensemble_prefix(), and
*            use_tensor_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name associated with the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param function The name of the function in the script to run
*   \param function_length The length of the function name string,
*                          excluding null terminating character
*   \param inputs The tensor keys of inputs tensors to use in the script
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs The tensor keys of output tensors that will be used
*                  to save script results
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \return Returns SRNoError on success or an error code on failure
*/
SRError run_script(void* c_client,
                   const char* name,
                   const size_t name_length,
                   const char* function,
                   const size_t function_length,
                   const char** inputs,
                   const size_t* input_lengths,
                   const size_t n_inputs,
                   const char** outputs,
                   const size_t* output_lengths,
                   const size_t n_outputs);
/*!
*   \brief Run a script function in the database using the specified input
*          and output tensors
*   \details The script key used to locate the script to be run
*            may be formed by applying a prefix to the supplied
*            name. Similarly, the tensor names in the
*            input and output arrays may be prefixed.
*            See set_data_source(), use_model_ensemble_prefix(), and
*            use_tensor_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name associated with the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param function The name of the function in the script to run
*   \param function_length The length of the function name string,
*                          excluding null terminating character
*   \param inputs The tensor keys of inputs tensors to use in the script
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs The tensor keys of output tensors that will be used
*                  to save script results
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \return Returns SRNoError on success or an error code on failure
*/
SRError run_script(void* c_client,
                   const char* name,
                   const size_t name_length,
                   const char* function,
                   const size_t function_length,
                   const char** inputs,
                   const size_t* input_lengths,
                   const size_t n_inputs,
                   const char** outputs,
                   const size_t* output_lengths,
                   const size_t n_outputs);

/*!
*   \brief Run a script function in the database using the specified input
*          and output tensors in a multi-GPU system
*   \details The script key used to locate the script to be run
*            may be formed by applying a prefix to the supplied
*            name. Similarly, the tensor names in the
*            input and output arrays may be prefixed.
*            See set_data_source(), use_model_ensemble_prefix(), and
*            use_tensor_ensemble_prefix() for more details
*   \param c_client The client object to use for communication
*   \param name The name associated with the script
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param function The name of the function in the script to run
*   \param function_length The length of the function name string,
*                          excluding null terminating character
*   \param inputs The tensor keys of inputs tensors to use in the script
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs The tensor keys of output tensors that will be used
*                  to save script results
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \param offset index of the current image, such as a processor
*                   ID or MPI rank
*   \param first_gpu the first GPU (zero-based) to use with the script
*   \param num_gpus the number of gpus for which the script was stored
*   \return Returns SRNoError on success or an error code on failure
*/
SRError run_script_multigpu(void* c_client,
                            const char* name,
                            const size_t name_length,
                            const char* function,
                            const size_t function_length,
                            const char** inputs,
                            const size_t* input_lengths,
                            const size_t n_inputs,
                            const char** outputs,
                            const size_t* output_lengths,
                            const size_t n_outputs,
                            const int offset,
                            const int first_gpu,
                            const int num_gpus);

/*!
*   \brief Check parameters for all parameters common to run_model methods
*   \details Make sure that all pointers are not void and that the size
*            of the inputs and outputs is not zero 
*   \param c_client The client object to use for communication
*   \param name The name to associate with the model
*   \param inputs One or more names of model input nodes (TF models only)
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes (TF models only)
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*/
void _check_params_run_model(void* c_client,
                  const char* name,
                  const char** inputs,
                  const size_t* input_lengths,
                  const size_t n_inputs,
                  const char** outputs,
                  const size_t* output_lengths,
                  const size_t n_outputs);

/*!
*   \brief Run a model in the database using the specified input and
*          output tensors
*   \details The model key used to locate the model to be run
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_model_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name associated with the model
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param inputs The names of inputs tensors to use in the script
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs The names of output tensors to be used
*                  to save script results
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \return Returns SRNoError on success or an error code on failure
*/
SRError run_model(void* c_client,
                  const char* name,
                  const size_t name_length,
                  const char** inputs,
                  const size_t* input_lengths,
                  const size_t n_inputs,
                  const char** outputs,
                  const size_t* output_lengths,
                  const size_t n_outputs);

/*!
*   \brief Run a model in the database using the specificed input and
*          output tensors in a multi-GPU system
*   \details The model key used to locate the model to be run
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_model_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name associated with the model
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param inputs The names of inputs tensors to use in the script
*   \param input_lengths The length of each input name string,
*                        excluding null terminating character
*   \param n_inputs The number of inputs
*   \param outputs The names of output tensors to be used
*                  to save script results
*   \param output_lengths The length of each output name string,
*                         excluding null terminating character
*   \param n_outputs The number of outputs
*   \param offset index of the current image, such as a processor
*                 ID or MPI rank
*   \param first_gpu the first GPU (zero-based) to use with the model
*   \param num_gpus the number of gpus for which the script was stored
*   \return Returns SRNoError on success or an error code on failure
*/
SRError run_model_multigpu(void* c_client,
                  const char* name,
                  const size_t name_length,
                  const char** inputs,
                  const size_t* input_lengths,
                  const size_t n_inputs,
                  const char** outputs,
                  const size_t* output_lengths,
                  const size_t n_outputs,
                  const int offset,
                  const int first_gpu,
                  const int num_gpus);

/*!
*   \brief Remove a model from the database
*   \param c_client The client object to use for communication
*   \param name The name associated with the model
*   \param name_length The length of the name string,
*   \return Returns SRNoError on success or an error code on failure
*/
SRError delete_model(void* c_client,
                     const char* name,
                     const size_t name_length);
/*!
*   \brief Remove a model from the database
*   \details The model key used to locate the model to be deleted
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source() and use_model_ensemble_prefix()
*            for more details.
*            The first_gpu and num_gpus parameters must match those used
*            when the model was stored.
*   \param c_client The client object to use for communication
*   \param name The name associated with the model
*   \param name_length The length of the name string,
*   \param first_gpu the first GPU (zero-based) to use with the model
*   \param num_gpus the number of gpus for which the model was stored
*   \return Returns SRNoError on success or an error code on failure
*/
SRError delete_model_multigpu(void* c_client,
                              const char* name,
                              const size_t name_length,
                              const int first_gpu,
                              const int num_gpus);

/*!
*   \brief Remove a script from the database
*   \param c_client The client object to use for communication
*   \param name The name associated with the script
*   \param name_length The length of the name string,
*   \return Returns SRNoError on success or an error code on failure
*/
SRError delete_script(void* c_client,
                      const char* name,
                      const size_t name_length);

/*!
*   \brief Remove a script from the database that was stored
*          for use with multiple GPUs
*   \details The script key used to locate the script to be deleted
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source() and use_model_ensemble_prefix()
*            for more details.
*            The first_gpu and num_gpus parameters must match those used
*            when the script was stored.
*   \param c_client The client object to use for communication
*   \param name The name associated with the model
*   \param name_length The length of the name string,
*   \param first_gpu the first GPU (zero-based) to use with the model
*   \param num_gpus the number of gpus for which the model was stored
*   \return Returns SRNoError on success or an error code on failure
*/
SRError delete_script_multigpu(void* c_client,
                            const char* name,
                            const size_t name_length,
                            const int first_gpu,
                            const int num_gpus);

/*!
*   \brief Check if a key exists in the database
*   \details The key to be checked is not prefixed in any way. If prefixing
*            is enabled, callers must manually prefix names to form
*            keys to be checked.
*   \param c_client The client object to use for communication
*   \param key The key that will be checked in the database
*   \param key_length The length of the key string,
*                     excluding null terminating character
*   \param exists Receives whether the key exists
*   \return Returns SRNoError on success or an error code on failure
*/
SRError key_exists(void* c_client,
                   const char* key,
                   const size_t key_length,
                   bool* exists);

/*!
*   \brief Check if a tensor exists in the database
*   \details The tensor key used to check for tensor existence
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_tensor_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the tensor that will be checked in the database.
*               The full tensor key corresponding to \p name will be formed
*               in accordance with the current prefixing behavior
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param exists Receives whether the tensor exists
*   \return Returns SRNoError on success or an error code on failure
*/
SRError tensor_exists(void* c_client,
                      const char* name,
                      const size_t name_length,
                      bool* exists);

/*!
*   \brief Check if a model or script exists in the database
*   \details The model or script key used to check for existence
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_model_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the entity that will be checked in the database.
*               The full model/script key corresponding to \p name will be
*               formed in accordance with the current prefixing behavior
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param exists Receives whether the model/script exists
*   \return Returns SRNoError on success or an error code on failure
*/
SRError model_exists(void* c_client,
                     const char* name,
                     const size_t name_length,
                     bool* exists);

/*!
*   \brief Check if a dataset exists in the database
*   \details The dataset key used to check for dataset existence
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_dataset_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the dataset that will be checked in the database.
*               The full key corresponding to \p name will be formed
*               in accordance with the current prefixing behavior
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param exists Receives whether the dataset exists
*   \return Returns SRNoError on success or an error code on failure
*/
SRError dataset_exists(void* c_client,
                       const char* name,
                       const size_t name_length,
                       bool* exists);

/*!
*   \brief Check if a key exists in the database, repeating the check
*          at a specified frequency and number of repetitions
*   \details The key to be checked is not prefixed in any way. If prefixing
*            is enabled, callers must manually prefix names to form
*            keys to be checked.
*   \param c_client The client object to use for communication
*   \param key The key to be checked in the database
*   \param key_length The length of the key string,
*                     excluding null terminating character
*   \param poll_frequency_ms The time delay between checks, in milliseconds
*   \param num_tries The total number of times to check for the key
*   \param exists Receives whether the key is found within the
*                 specified number of tries
*   \return Returns SRNoError on success or an error code on failure
*/
SRError poll_key(void* c_client,
                 const char* key,
                 const size_t key_length,
                 const int poll_frequency_ms,
                 const int num_tries,
                 bool* exists);

/*!
*   \brief Check if a model or script exists in the database, repeating the
*          check at a specified frequency and number of repetitions
*   \details The model or script key used to check for existence
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_model_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the entity to be checked in the database.
*               The full key associated to \p name will be formed according
*               to the prefixing behavior
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param poll_frequency_ms The time delay between checks, in milliseconds
*   \param num_tries The total number of times to check for the model key
*   \param exists Receives whether the model is found within the
*                 specified number of tries
*   \return Returns SRNoError on success or an error code on failure
*/
SRError poll_model(void* c_client,
                   const char* name,
                   const size_t name_length,
                   const int poll_frequency_ms,
                   const int num_tries,
                   bool* exists);

/*!
*   \brief Check if a tensor exists in the database, repeating the check
*          at a specified frequency and number of repetitions
*   \details The tensor key used to check for tensor existence
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_tensor_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the entity to be checked in the database.
*               The full key associated to \p name will be formed according
*               to the prefixing behavior
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param poll_frequency_ms The time delay between checks, in milliseconds
*   \param num_tries The total number of times to check for the tensor key
*   \param exists Receives whether the tensor is found within the
*                 specified number of tries
*   \return Returns SRNoError on success or an error code on failure
*/
SRError poll_tensor(void* c_client,
                    const char* name,
                    const size_t name_length,
                    const int poll_frequency_ms,
                    const int num_tries,
                    bool* exists);

/*!
*   \brief Check if a dataset exists in the database, repeating the check
*          at a specified frequency and number of repetitions
*   \details The dataset key used to check for dataset existence
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()
*            and use_dataset_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the entity to be checked in the database.
*               The full key associated to \p name will be formed according
*               to the prefixing behavior
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param poll_frequency_ms The time delay between checks, in milliseconds
*   \param num_tries The total number of times to check for the dataset key
*   \param exists Receives whether the tensor is found within the
*                 specified number of tries
*   \return Returns sr_ok on success
*/
SRError poll_dataset(void* c_client,
                     const char* name,
                     const size_t name_length,
                     const int poll_frequency_ms,
                     const int num_tries,
                     bool* exists);

/*!
*   \brief Set the data source, a key prefix for future read operations
*   \details When running multiple applications, such as an ensemble
*            computation, there is a risk that the same name is used
*            for a tensor, dataset, script, or model by more than one
*            executing entity. In order to prevent this sort of collision,
*            SmartRedis affords the ability to add a prefix to names,
*            thereby associating them with the name of the specific
*            entity that the prefix corresponds to. For writes to
*            the database when prefixing is activated, the prefix
*            used is taken from the SSKEYOUT environment variable.
*            For reads from the database, the default is to use the
*            first prefix from SSKEYIN. If this is the same as the
*            prefix from SSKEYOUT, the entity will read back the
*            same data it wrote; however, this function allows an entity
*            to read from data written by another entity (i.e. use
*            the other entity's key.)
*   \param c_client The client object to use for communication
*   \param source_id The prefix for read operations; must have
*          previously been set via the SSKEYIN environment variable
*   \param source_id_length The length of the source_id string,
*                           excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError set_data_source(void* c_client,
                        const char* source_id,
                        const size_t source_id_length);

/*!
*   \brief Control whether tensor names are prefixed (e.g. in an
*          ensemble) when forming database keys
*   \details This function can be used to avoid key collisions in an
*            ensemble by prepending the string value from the
*            environment variable SSKEYIN to tensor names.
*            Prefixes will only be used if they were previously set through
*            Keys for entities created before this function is called
*            the environment variables SSKEYOUT and SSKEYIN.
*            will not be retroactively prefixed.
*            By default, the client prefixes tensor keys
*            with the first prefix specified with the SSKEYIN
*            and SSKEYOUT environment variables.
*
*   \param c_client The client object to use for communication
*   \param use_prefix If true, all future operations on tensors and
*                     datasets will use a prefix, if available.
*   \return Returns SRNoError on success or an error code on failure
*/
SRError use_tensor_ensemble_prefix(void* c_client, bool use_prefix);

/*!
*   \brief Control whether dataset names are prefixed (e.g. in an
*          ensemble) when forming database keys
*   \details This function can be used to avoid key collisions in an
*            ensemble by prepending the string value from the
*            environment variable SSKEYIN to tensor and dataset names.
*            Prefixes will only be used if they were previously set through
*            Keys for entities created before this function is called
*            the environment variables SSKEYOUT and SSKEYIN.
*            will not be retroactively prefixed.
*            By default, the client prefixes dataset keys
*            with the first prefix specified with the SSKEYIN
*            and SSKEYOUT environment variables.
*
*   \param c_client The client object to use for communication
*   \param use_prefix If true, all future operations on tensors and
*                     datasets will use a prefix, if available.
*   \return Returns SRNoError on success or an error code on failure
*/
SRError use_dataset_ensemble_prefix(void* c_client, bool use_prefix);

/*!
*   \brief Control whether model and script names are
*          prefixed (e.g. in an ensemble) when forming database keys
*   \details This function can be used to avoid key collisions in an
*            ensemble by prepending the string value from the
*            environment variable SSKEYIN to model and script names.
*            Prefixes will only be used if they were previously set through
*            Keys for entities created before this function is called
*            the environment variables SSKEYOUT and SSKEYIN.
*            will not be retroactively prefixed.
*            By default, the client does not prefix model and script keys.
*
*   \param c_client The client object to use for communication
*   \param use_prefix If set to true, all future operations on models and
*                     scripts will use a prefix, if available.
*   \return Returns SRNoError on success or an error code on failure
*/
SRError use_model_ensemble_prefix(void* c_client, bool use_prefix);

/*!
<<<<<<< HEAD
*   \brief Control whether aggregation lists are prefixed
*   \details This function can be used to avoid key collisions in an
*            ensemble by prepending the string value from the
*            environment variable SSKEYIN and/or SSKEYOUT to
*            aggregation list names.  Prefixes will only be used if
*            they were previously set through the environment variables
*            SSKEYOUT and SSKEYIN. Keys for aggregation lists created
*            before this function is called will not be retroactively
*            prefixed. By default, the client prefixes aggregation
*            list keys with the first prefix specified with the SSKEYIN
*            and SSKEYOUT environment variables.  Note that
*            use_dataset_ensemble_prefix() controls prefixing
*            for the entities in the aggregation list, and
*            use_dataset_ensemble_prefix() should be given the
*            same value that was used during the initial
*            setting of the DataSet into the database.
*   \param c_client The client object to use for communication
*   \param use_prefix If set to true, all future operations
*                    on aggregation lists will use
*                    a prefix, if available.
*   \return Returns SRNoError on success or an error code on failure
*/
SRError use_list_ensemble_prefix(void* c_client, bool use_prefix);

/*!
*   \brief Appends a dataset to the aggregation list
*   \details When appending a dataset to an aggregation list,
*            the list will automatically be created if it does not
*            exist (i.e. this is the first entry in the list).
*            Aggregation lists work by referencing the dataset
*            by storing its key, so appending a dataset
*            to an aggregation list does not create a copy of the
*            dataset.  Also, for this reason, the dataset
*            must have been previously placed into the database
*            with a separate call to put_dataset().
*   \param c_client The client object to use for communication
*   \param list_name The name of the aggregation list
*   \param list_name_length The size in characters of the list name,
*                           including null terminator
*   \param dataset The DataSet to append
*   \return Returns SRNoError on success or an error code on failure
*/
SRError append_to_list(void* c_client, const char* list_name,
                       const size_t list_name_length, const void* dataset);

/*!
*   \brief Delete an aggregation list
*   \details The key used to locate the aggregation list to be
*            deleted may be formed by applying a prefix to the
*            supplied name. See set_data_source()
*            and use_list_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param list_name The name of the aggregation list
*   \param list_name_length The size in characters of the list name,
*                           including null terminator
*   \return Returns SRNoError on success or an error code on failure
*/
SRError delete_list(void* c_client, const char* list_name,
                    const size_t list_name_length);

/*!
*   \brief Copy an aggregation list
*   \details The source and destination aggregation list keys used to
*            locate and store the aggregation list may be formed by
*            applying prefixes to the supplied src_name and dest_name.
*            See set_data_source() and use_list_ensemble_prefix()
*            for more details.
*   \param c_client The client object to use for communication
*   \param src_name The source list name
*   \param src_name_length The size in characters of the source list name,
*                          including null terminator
*   \param dest_name The destination list name
*   \param dest_name_length The size in characters of the destination list name,
*                           including null terminator
*   \return Returns SRNoError on success or an error code on failure
*/
SRError copy_list(void* c_client,
                  const char* src_name, const size_t src_name_length,
                  const char* dest_name, const size_t dest_name_length);

/*!
*   \brief Rename an aggregation list
*   \details The initial and target aggregation list key used to find and
*            relocate the list may be formed by applying prefixes to
*            the supplied src_name and dest_name. See set_data_source()
*            and use_list_ensemble_prefix() for more details.
*   \param c_client The client object to use for communication
*   \param src_name The initial list name
*   \param src_name_length The size in characters of the initial list name,
*                          including null terminator
*   \param dest_name The target list name
*   \param dest_name_length The size in characters of the target list name,
*                           including null terminator
*   \return Returns SRNoError on success or an error code on failure
*/
SRError rename_list(void* c_client,
                    const char* src_name, const size_t src_name_length,
                    const char* dest_name, const size_t dest_name_length);

/*!
*   \brief Get the number of entries in the list
*   \param c_client The client object to use for communication
*   \param list_name The list name
*   \param list_name_length The size in characters of the list name,
*                           including null terminator
*   \param result_length Receives the length of the list
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_list_length(void* c_client, const char* list_name,
                        const size_t list_name_length, int* result_length);

/*!
*   \brief Poll list length until length is equal
*          to the provided length.  If maximum number of
*          attempts is exceeded, false is returned.
*   \details The aggregation list key used to check for list length
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source() and use_list_ensemble_prefix()
*            for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the list
*   \param name_length The size in characters of the list name,
*                      including null terminator
*   \param list_length The desired length of the list
*   \param poll_frequency_ms The time delay between checks,
*                            in milliseconds
*   \param num_tries The total number of times to check for the name
*   \param poll_result Receives the result of the poll:
*                      true if the list is found with a length greater
*                      than or equal to the provided length, otherwise false
*   \return Returns SRNoError on success or an error code on failure
*/
SRError poll_list_length(void* c_client, const char* name,
                         const size_t name_length, int list_length,
                         int poll_frequency_ms, int num_tries, bool* poll_result);

/*!
*   \brief Poll list length until length is greater than or equal
*          to the user-provided length. If maximum number of
*          attempts is exceeded, false is returned.
*   \details The aggregation list key used to check for list length
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source() and use_list_ensemble_prefix()
*            for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the list
*   \param name_length The size in characters of the list name,
*                      including null terminator
*   \param list_length The desired length of the list
*   \param poll_frequency_ms The time delay between checks,
*                            in milliseconds
*   \param num_tries The total number of times to check for the name
*   \param poll_result Receives the result of the poll:
*                      true if the list is found with a length greater
*                      than or equal to the provided length, otherwise false
*   \return Returns SRNoError on success or an error code on failure
*/
SRError poll_list_length_gte(void* c_client, const char* name,
                             const size_t name_length, int list_length,
                             int poll_frequency_ms, int num_tries,
                             bool* poll_result);

/*!
*   \brief Poll list length until length is less than or equal
*          to the user-provided length. If maximum number of
*          attempts is exceeded, false is returned.
*   \details The aggregation list key used to check for list length
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source() and use_list_ensemble_prefix()
*            for more details.
*   \param c_client The client object to use for communication
*   \param name The name of the list
*   \param name_length The size in characters of the list name,
*                      including null terminator
*   \param list_length The desired length of the list
*   \param poll_frequency_ms The time delay between checks,
*                            in milliseconds
*   \param num_tries The total number of times to check for the name
*   \param poll_result Receives the result of the poll:
*                      true if the list is found with a length less
*                      than or equal to the provided length, otherwise false
*   \return Returns SRNoError on success or an error code on failure
*/
SRError poll_list_length_lte(void* c_client, const char* name,
                             const size_t name_length, int list_length,
                             int poll_frequency_ms, int num_tries,
                             bool* poll_result);

/*!
*   \brief Get datasets from an aggregation list
*   \details The aggregation list key used to retrieve datasets
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source() and use_list_ensemble_prefix()
*            for more details.  An empty or nonexistant
*            aggregation list returns an empty vector.
*   \param c_client The client object to use for communication
*   \param list_name The name of the aggregation list
*   \param list_name_length The size in characters of the list name,
*                           including null terminator
*   \param datasets Receives an array of datasets included in the list
*   \param num_datasets Receives the number of datasets returned
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_datasets_from_list(void* c_client, const char* list_name,
                               const size_t list_name_length,
                               void*** datasets, size_t* num_datasets);

/*!
*   \brief Get a range of datasets (by index) from an aggregation list
*   \details The aggregation list key used to retrieve datasets
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()  and use_list_ensemble_prefix()
*            for more details.  An empty or nonexistant aggregation
*            list returns an empty vector.  If the provided
*            end_index is beyond the end of the list, that index will
*            be treated as the last index of the list.  If start_index
*            and end_index are inconsistent (e.g. end_index is less
*            than start_index), an empty list of datasets will be returned.
*   \param c_client The client object to use for communication
*   \param list_name The name of the aggregation list
*   \param list_name_length The size in characters of the list name,
*                           including null terminator
*   \param start_index The starting index of the range (inclusive,
*                      starting at zero).  Negative values are
*                      supported.  A negative value indicates offsets
*                      starting at the end of the list. For example, -1 is
*                      the last element of the list.
*   \param end_index The ending index of the range (inclusive,
*                    starting at zero).  Negative values are
*                    supported.  A negative value indicates offsets
*                    starting at the end of the list. For example, -1 is
*                    the last element of the list.
*   \param datasets Receives an array of datasets included in the list
*   \param num_datasets Receives the number of datasets returned
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_dataset_list_range(
    void* c_client,
    const char* list_name,
    const size_t list_name_length,
    const int start_index,
    const int end_index,
    void*** datasets,
    size_t* num_datasets);

/*!
*   \brief Get a range of datasets (by index) from an aggregation list and
           copy them into an already allocated vector of datasets. Note,
           while this method could be used by C clients, its primary
           use case is for the Fortran client.
*   \details The aggregation list key used to retrieve datasets
*            may be formed by applying a prefix to the supplied
*            name. See set_data_source()  and use_list_ensemble_prefix()
*            for more details.  An empty or nonexistant aggregation
*            list returns an empty vector.  If the provided
*            end_index is beyond the end of the list, that index will
*            be treated as the last index of the list.  If start_index
*            and end_index are inconsistent (e.g. end_index is less
*            than start_index), an empty list of datasets will be returned.
*   \param c_client The client object to use for communication
*   \param list_name The name of the aggregation list
*   \param list_name_length The size in characters of the list name,
*                           including null terminator
*   \param start_index The starting index of the range (inclusive,
*                      starting at zero).  Negative values are
*                      supported.  A negative value indicates offsets
*                      starting at the end of the list. For example, -1 is
*                      the last element of the list.
*   \param end_index The ending index of the range (inclusive,
*                    starting at zero).  Negative values are
*                    supported.  A negative value indicates offsets
*                    starting at the end of the list. For example, -1 is
*                    the last element of the list.
*   \param datasets Receives an array of datasets included in the list
*   \return Returns SRNoError on success or an error code on failure
*/
SRError _get_dataset_list_range_allocated(
    void* c_client,
    const char* list_name,
    const size_t list_name_length,
    const int start_index,
    const int end_index,
    void** datasets);


/*!
*   \brief Retrieve a string representation of the client
*   \param c_client The client object to use for communication
*   \return A string with either the client representation or an error message
*/
const char* client_to_string(void* c_client);

#ifdef __cplusplus
}

#endif
#endif // SMARTREDIS_C_CLIENT_H
