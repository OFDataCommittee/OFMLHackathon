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

#ifndef SMARTREDIS_C_CLIENT_H
#define SMARTREDIS_C_CLIENT_H
///@file
///\brief C-wrappers for the C++ Client class
#include <stdlib.h>
#include <stdbool.h>
#include "client.h"
#include "enums/c_memory_layout.h"
#include "enums/c_tensor_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
*   \brief C-client constructor
*   \param cluster Flag to indicate if a
*                  database cluster is being used
*   \return Returns NULL on failure
*/
void* SmartRedisCClient(bool cluster);

/*!
*   \brief C-client destructor
*   \param c_client A pointer to the c-client
*                   to destroy
*/
void DeleteCClient(void* c_client);

/*!
*   \brief Put a DataSet object into the database
*   \param c_client A pointer to c client
*                   to use for communication
*   \param dataset The DataSet object to send
*/
void put_dataset(void* c_client, void* dataset);

/*!
*   \brief Get a DataSet object from the database
*   \param c_client A pointer to c client
*                   to use for communication
*   \param name The name of the dataset object to fetch
*   \param name_length The length of the name c-string,
*                      excluding null terminating character
*   \return Returns NULL on bad parameters
*/
void* get_dataset(void* c_client,
                  const char* name,
                  const size_t name_length);

/*!
*   \brief Move a DataSet to a new key
*   \param c_client A pointer to c client
*                   to use for communication
*   \param name The name of the dataset object
*   \param name_length The length of the name c-string,
*                      excluding null terminating character
*   \param new_name The new name of the dataset object
*   \param new_name_length The length of the new name
*                          c-string, excluding null
*                          terminating character
*/
void rename_dataset(void* c_client,
                    const char* name,
                    const size_t name_length,
                    const char* new_name,
                    const size_t new_name_length);

/*!
*   \brief Copy a DataSet to a new key
*   \param c_client A pointer to c client
*                   to use for communication
*   \param src_name The source name of the dataset object
*   \param src_name_length The length of the src_name c-string,
*                          excluding null terminating character
*   \param dest_name The destination name of the dataset object
*   \param dest_name_length The length of the dest_name c-string,
*                           excluding null terminating character
*/
void copy_dataset(void* c_client,
                  const char* src_name,
                  const size_t src_name_length,
                  const char* dest_name,
                  const size_t dest_name_length);

/*!
*   \brief Delete a DataSet
*   \param c_client A pointer to c client
*                   to use for communication
*   \param name The name of the dataset object
*   \param name_length The length of the name c-string,
*                      excluding null terminating character
*/
void delete_dataset(void* c_client,
                    const char* name,
                    const size_t name_length);

/*!
*   \brief Put a tensor into the database
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to use to place the tensor
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param data A c ptr to the beginning of the data
*   \param dims Length along each dimension of the tensor
*   \param n_dims The number of dimensions of the tensor
*   \param type The data type of the tensor
*   \param mem_layout The memory layout of the data
*/
void put_tensor(void* c_client,
                const char* key,
                const size_t key_length,
                void* data,
                const size_t* dims,
                const size_t n_dims,
                CTensorType type,
                CMemoryLayout mem_layout);

/*!
*   \brief Get a tensor from the database.  This method
*          will allocate memory for the tensor data and
*          dimensions.  This memory will be valid until
*          the c-client is destroyed.  The number
*          of dimensions and the tensor type will be
*          set based on the tensor retrieved from the
*          database.  The requested memory layout
*          will be used to shape the returned memory
*          space pointed to by the data pointer.
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to use to fetch the tensor
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param data A pointer to a c-ptr that will be set
*               to the newly allocated memory space.
*   \param dims A pointer to a size_t pointer that will
*                be pointed to newly allocated memory
*                that holds the dimensions of the tensor.
*   \param n_dims The number of dimensions of the tensor
*   \param type The data type of the tensor that is set
*               by the c-client
*   \param mem_layout The memory layout requested for the
*                     allocated memory space
*/
void get_tensor(void* c_client,
                const char* key,
                const size_t key_length,
                void** data,
                size_t** dims,
                size_t* n_dims,
                CTensorType* type,
                CMemoryLayout mem_layout);
/*!
*   \brief Get a tensor from the database and
*          fill the provided memory space
*          (result) that is layed out as defined by
*          dims
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to use to fetch the tensor
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param result A c ptr to the beginning of the
*                 memory space to fill.
*   \param dims The dimensions of the provided memory
*               space.
*   \param n_dims The number of dimensions of the tensor
*   \param type The data type of the provided memory
*               space.
*   \param mem_layout The memory layout of the provided
*                     memory space.
*/
void unpack_tensor(void* c_client,
                  const char* key,
                  const size_t key_length,
                  void* result,
                  const size_t* dims,
                  const size_t n_dims,
                  CTensorType type,
                  CMemoryLayout mem_layout);

/*!
*   \brief Move a tensor to a new key
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to use to fetch the tensor
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param new_key The new tensor key
*   \param new_key_length The length of the new_key c-string,
                          excluding null terminating characters
*/
void rename_tensor(void* c_client,
                   const char* key,
                   const size_t key_length,
                   const char* new_key,
                   const size_t new_key_length);

/*!
*   \brief Delete a tensor
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key of the tensor to delete
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*/
void delete_tensor(void* c_client,
                   const char* key,
                   const size_t key_length);

/*!
*   \brief  This method will copy a
*           tensor to the destination key
*   \param c_client A pointer to c client
*                   to use for communication
*   \param src_name The source name of the tensor
*   \param src_name_length The length of the src_name
*                          c-string, excluding null
*                          terminating character
*   \param dest_name The destination name of the tensor
*   \param dest_name_length The length of the dest_name
*                           c-string, excluding null
*                           terminating character
*/
void copy_tensor(void* c_client,
                 const char* src_name,
                 const size_t src_name_length,
                 const char* dest_name,
                 const size_t dest_name_length);

/*!
*   \brief  Set a model (from file)
*           in the database for future execution
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to associate with the model
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param model_file The source file for the model
*   \param model_file_length The length of the model_file c-string,
*                            excluding null terminating character
*   \param backend The name of the backend
*                  (TF, TFLITE, TORCH, ONNX)
*   \param backend_length The length of the backend c-string,
*                        excluding null terminating character
*   \param device The name of the device for execution
*                 (e.g. CPU or GPU)
*   \param device_length The length of the device c-string,
*                        excluding null terminating character
*   \param batch_size The batch size for model execution
*   \param min_batch_size The minimum batch size for model
*                         execution
*   \param tag A tag to attach to the model for
*              information purposes
*   \param tag_length The length of the tag c-string,
*                     excluding null terminating character
*   \param inputs One or more names of model input nodes
*                 (TF models only)
*   \param input_lengths The length of each input name
*                        c-string, excluding null
*                        terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes
*                 (TF models only)
*   \param output_lengths The length of each output name
*                         c-string, excluding null terminating
*                         character
*   \param n_outputs The number of outputs
*/
void set_model_from_file(void* c_client,
                         const char* key,
                         const size_t key_length,
                         const char* model_file,
                         const size_t model_file_length,
                         const char* backend,
                         const size_t backend_length,
                         const char* device,
                         const size_t device_length,
                         const int batch_size,
                         const int min_batch_size,
                         const char* tag,
                         const size_t tag_length,
                         const char** inputs,
                         const size_t* input_lengths,
                         const size_t n_inputs,
                         const char** outputs,
                         const size_t* output_lengths,
                         const size_t n_outputs);

/*!
*   \brief  Set a model (from buffer)
*           in the database for future execution
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to associate with the model
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param model The model as a continuous buffer
*   \param model_length The length of the model c-string,
*                            excluding null terminating character
*   \param backend The name of the backend
*                  (TF, TFLITE, TORCH, ONNX)
*   \param backend_length The length of the backend c-string,
*                        excluding null terminating character
*   \param device The name of the device for execution
*                 (e.g. CPU or GPU)
*   \param device_length The length of the device c-string,
*                        excluding null terminating character
*   \param batch_size The batch size for model execution
*   \param min_batch_size The minimum batch size for model
*                         execution
*   \param tag A tag to attach to the model for
*              information purposes
*   \param tag_length The length of the tag c-string,
*                     excluding null terminating character
*   \param inputs One or more names of model input nodes
*                 (TF models only)
*   \param input_lengths The length of each input name
*                        c-string, excluding null
*                        terminating character
*   \param n_inputs The number of inputs
*   \param outputs One or more names of model output nodes
*                 (TF models only)
*   \param output_lengths The length of each output name
*                         c-string, excluding null terminating
*                         character
*   \param n_outputs The number of outputs
*/
void set_model(void* c_client,
               const char* key,
               const size_t key_length,
               const char* model,
               const size_t model_length,
               const char* backend,
               const size_t backend_length,
               const char* device,
               const size_t device_length,
               const int batch_size,
               const int min_batch_size,
               const char* tag,
               const size_t tag_length,
               const char** inputs,
               const size_t* input_lengths,
               const size_t n_inputs,
               const char** outputs,
               const size_t* output_lengths,
               const size_t n_outputs);

/*!
*   \brief Get a model in the database
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to use to get the model
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param model_length The length of the model buffer
*                       c-string, excluding null
*                       terminating character
*   \returns The model as a c-string or NULL on bad parameters
*/
const char* get_model(void* c_client,
                     const char* key,
                     const size_t key_length,
                     size_t* model_length);

/*!
*   \brief Set a script from file in the
*          database for future execution
*   \param key The key to associate with the script
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param device The name of the device for execution
*                 (e.g. CPU or GPU)
*   \param device_length The length of the device name
*                        c-string, excluding null
*                        terminating character
*   \param script_file The source file for the script
*   \param script_file_length The length of the script
*                             file name c-string, excluding
*                             null terminating character
*/
void set_script_from_file(void* c_client,
                          const char* key,
                          const size_t key_length,
                          const char* device,
                          const size_t device_length,
                          const char* script_file,
                          const size_t script_file_length);

/*!
*   \brief Set a script (from buffer)
*          in the database for future execution
*   \param key The key to associate with the script
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param device The name of the device for execution
*                 (e.g. CPU or GPU)
*   \param device_length The length of the device name
*                        c-string, excluding null
*                        terminating character
*   \param script The script as a c-string buffer
*   \param script_length The length of the script
*                        c-string, excluding
*                        null terminating character
*/
void set_script(void* c_client,
                const char* key,
                const size_t key_length,
                const char* device,
                const size_t device_length,
                const char* script,
                const size_t script_length);

/*!
*   \brief Get a script in the database.  The
*          memory associated with the script
*          c-str is valid until the client is
*          destroyed.
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key to use to get the script
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param script A pointer that is pointed to newly
*                 allocated memory containing the script
*   \param script_length The length of the script buffer
*                        c-string, excluding null
*                        terminating character
*/
void get_script(void* c_client,
                const char* key,
                const size_t key_length,
                const char** script,
                size_t* script_length);

/*!
*   \brief Run a script function in the database using the
*          specificed input and output tensors
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key associated with the script
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param function The name of the function in the script to run
*   \param function_length The length of the function name c-string,
*                          excluding null terminating character
*   \param inputs The keys of inputs tensors to use
*                 in the script
*   \param input_lengths The length of each input name
*                        c-string, excluding null terminating
*                        character
*   \param n_inputs The number of inputs
*   \param outputs The keys of output tensors that
*                  will be used to save script results
*   \param output_lengths The length of each output name
*                         c-string, excluding null terminating
*                         character
*   \param n_outputs The number of outputs
*/
void run_script(void* c_client,
                const char* key,
                const size_t key_length,
                const char* function,
                const size_t function_length,
                const char** inputs,
                const size_t* input_lengths,
                const size_t n_inputs,
                const char** outputs,
                const size_t* output_lengths,
                const size_t n_outputs);

/*!
*   \brief Run a model in the database using the
*          specificed input and output tensors
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key associated with the model
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param inputs The keys of inputs tensors to use
*                 in the script
*   \param input_lengths The length of each input name
*                        c-string, excluding null terminating
*                        character
*   \param n_inputs The number of inputs
*   \param outputs The keys of output tensors that
*                  will be used to save script results
*   \param output_lengths The length of each output name
*                         c-string, excluding null terminating
*                         character
*   \param n_outputs The number of outputs
*/
void run_model(void* c_client,
               const char* key,
               const size_t key_length,
               const char** inputs,
               const size_t* input_lengths,
               const size_t n_inputs,
               const char** outputs,
               const size_t* output_lengths,
               const size_t n_outputs);

/*!
*   \brief Check if the key exists in the database
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key that will be checked in the database
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \returns Returns true if the key exists in the database
*/
bool key_exists(void* c_client,
                const char* key,
                const size_t key_length);

/*!
*   \brief Check if a tensor or dataset exists in the database
*   \param c_client A pointer to c client
*                   to use for communication
*   \param name The name of the entity that will be checked 
*               in the database. The full key associated to
*               \p name will formed according to the
*               prefixing behavior
*   \param name_length The length of the name c-string,
*                      excluding null terminating character
*   \returns Returns true if the key exists in the database
*/
bool tensor_exists(void* c_client,
                   const char* name,
                   const size_t name_length);

/*!
*   \brief Check if a model or script exists in the database
*   \param c_client A pointer to c client
*                   to use for communication
*   \param name The name of the entity that will be checked 
*               in the database. The full key associated to
*               \p name will formed according to the
*               prefixing behavior
*   \param name_length The length of the name c-string,
*                      excluding null terminating character
*   \returns Returns true if the key exists in the database
*/
bool model_exists(void* c_client,
                  const char* name,
                  const size_t name_length);

/*!
*   \brief Check if the key exists in the database at a
*          specified frequency for a specified number
*          of times
*   \param c_client A pointer to c client
*                   to use for communication
*   \param key The key that will be checked in the database
*   \param key_length The length of the key c-string,
*                     excluding null terminating character
*   \param poll_frequency_ms The frequency of checks for the
*                            key in milliseconds
*   \param num_tries The total number of times to check for
*                    the specified number of keys.  If the
*                    value is set to -1, the key will be
*                    polled indefinitely.
*   \returns Returns true if the key is found within the
*            specified number of tries, otherwise false.
*/
bool poll_key(void* c_client,
              const char* key,
              const size_t key_length,
              const int poll_frequency_ms,
              const int num_tries);

/*!
*   \brief Check if a model or script exists in the database at a
*          specified frequency for a specified number
*          of times
*   \param c_client A pointer to c client
*                   to use for communication
*   \param name The name of the entity that will be checked 
*               in the database. The full key associated to
*               \p name will formed according to the
*               prefixing behavior
*   \param name_length The length of the name c-string,
*                     excluding null terminating character
*   \param poll_frequency_ms The frequency of checks for the
*                            key in milliseconds
*   \param num_tries The total number of times to check for
*                    the specified number of keys.  If the
*                    value is set to -1, the key will be
*                    polled indefinitely.
*   \returns Returns true if the key is found within the
*            specified number of tries, otherwise false.
*/
bool poll_model(void* c_client,
                const char* name,
                const size_t name_length,
                const int poll_frequency_ms,
                const int num_tries);

/*!
*   \brief Check if a tensor or dataset exists in the database at a
*          specified frequency for a specified number
*          of times
*   \param c_client A pointer to c client
*                   to use for communication
*   \param name The name of the entity that will be checked 
*               in the database. The full key associated to
*               \p name will formed according to the
*               prefixing behavior
*   \param name_length The length of the name c-string,
*                     excluding null terminating character
*   \param poll_frequency_ms The frequency of checks for the
*                            key in milliseconds
*   \param num_tries The total number of times to check for
*                    the specified number of keys.  If the
*                    value is set to -1, the key will be
*                    polled indefinitely.
*   \returns Returns true if the key is found within the
*            specified number of tries, otherwise false.
*/
bool poll_tensor(void* c_client,
                 const char* name,
                 const size_t name_length,
                 const int poll_frequency_ms,
                 const int num_tries);

/*!
*   \brief Set the data source (i.e. key prefix for
*          get functions)
*   \param source_id The prefix for retrieval commands
*   \param source_id_length The length of the source_id c-string,
*                           excluding null terminating character
*/
void set_data_source(void* c_client,
                     const char* source_id,
                     const size_t source_id_length);

/*!
* \brief Set whether names of tensor and dataset entities should be
*        prefixed (e.g. in an ensemble) to form database keys.
*        Prefixes will only be used if they were previously set through
*        the environment variables SSKEYOUT and SSKEYIN.
*        Keys of entities created before this function is called 
*        will not be affected.
*        By default, the client prefixes tensor and dataset keys
*        with the first prefix specified with the SSKEYIN
*        and SSKEYOUT environment variables.
*
* \param use_prefix If set to true, all future operations
*                   on tensors and datasets will use 
*                   a prefix, if available.
*/
void use_tensor_ensemble_prefix(void* c_client, bool use_prefix);

/*!
* \brief Set whether names of model and script entities should be
*        prefixed (e.g. in an ensemble) to form database keys.
*        Prefixes will only be used if they were previously set through
*        the environment variables SSKEYOUT and SSKEYIN.
*        Keys of entities created before this function is called 
*        will not be affected.
*        By default, the client does not prefix model and script keys.
*
* \param use_prefix If set to true, all future operations
*                   on models and scripts will use 
*                   a prefix, if available.
*/
void use_model_ensemble_prefix(void* c_client, bool use_prefix);

#ifdef __cplusplus
}

#endif
#endif // SMARTREDIS_C_CLIENT_H
