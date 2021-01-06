//#ifndef SMARTSIM_CPP_CLIENT_H
#define SMARTSIM_CPP_CLIENT_H
#ifdef __cplusplus
#include "string.h"
#include "stdlib.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <sw/redis++/redis++.h>
#include <fstream>
#include "dataset.h"
#include "command.h"
#include "commandlist.h"
#include "tensorbase.h"
#include "tensor.h"
#include "dbnode.h"
#include "memorylist.h"
#include "commandreply.h"
#include "enums/cpp_tensor_type.h"
#include "enums/cpp_memory_layout.h"

///@file
///\brief The C++ Client class

namespace SILC {

class Client;

typedef redisReply ReplyElem;

class Client
{
public:
  //! Client constructor
  Client(
      bool cluster /*!< Flag to indicate if a database cluster is being used*/,
      bool fortran_array = false /*!< Flag to indicate if fortran arrays are being used*/
      );

  //! Client destructor
  ~Client();

  //! Put a DataSet object into the database
  void put_dataset(DataSet& dataset /*!< The DataSet object to send*/
                   );

  //! Get a DataSet object from the database
  DataSet get_dataset(const std::string& name /*!< The name of the dataset object to fetch*/
                      );

  //! Move a DataSet to a new key
  void rename_dataset(const std::string& name /*!< The name of the dataset object*/,
                      const std::string& new_name /*!< The name of the dataset object*/);

  //! Copy a DataSet to a new key
  void copy_dataset(const std::string& src_name /*!< The source name of the dataset object*/,
                    const std::string& dest_name /*!< The destination name of the dataset object*/
                    );

  //! Delete a DataSet
  void delete_dataset(const std::string& name /*!< The name of the dataset object*/
                      );

  //! Put a tensor into the database
  void put_tensor(const std::string& key /*!< The key to use to place the tensor*/,
                  void* data /*!< A c ptr to the beginning of the data*/,
                  const std::vector<size_t>& dims /*!< The dimensions of the tensor*/,
                  const TensorType type /*!< The data type of the tensor*/,
                  const MemoryLayout mem_layout /*! The memory layout of the data*/
                  );

  //! Get tensor data and return an allocated multi-dimensional array
  void get_tensor(const std::string& key /*!< The name used to reference the tensor*/,
                  void*& data /*!< A c_ptr to the tensor data */,
                  std::vector<size_t>& dims /*!< The dimensions of the provided array which should match the tensor*/,
                  TensorType& type /*!< The data type of the tensor*/,
                  const MemoryLayout mem_layout /*! The memory layout of the data*/
                  );

  //! Get tensor data and return an allocated multi-dimensional array (c-style interface for type and dimensions)
  void get_tensor(const std::string& key /*!< The name used to reference the tensor*/,
                  void*& data /*!< A c_ptr to the tensor data */,
                  size_t*& dims /*! The dimensions of the tensor retrieved*/,
                  size_t& n_dims /*! The number of dimensions retrieved*/,
                  TensorType& type /*! The number of dimensions retrieved*/,
                  const MemoryLayout mem_layout /*! The memory layout of the data*/
                  );

  //! Get tensor data and fill an already allocated array memory space
  void unpack_tensor(const std::string& key /*!< The key to use to fetch the tensor*/,
                     void* data /*!< A c ptr to the beginning of the result array to fill*/,
                     const std::vector<size_t>& dims /*!< The dimensions of the provided array which should match the tensor*/,
                     const TensorType type /*!< The data type corresponding the the user provided memory space*/,
                     const MemoryLayout mem_layout /*! The memory layout of the data*/
                     );

  //! Move a tensor to a new key
  void rename_tensor(const std::string& key /*!< The original tensor key*/,
                     const std::string& new_key /*!< The new tensor key*/);

  //! Delete a tensor
  void delete_tensor(const std::string& key /*!< The key of tensor to delete*/);

  //! This method will copy a tensor to the destination key
  void copy_tensor(const std::string& src_key /*!< The source tensor key*/,
                   const std::string& dest_key /*!< The destination tensor key*/
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

  //! Get a model in the database
  std::string_view get_model(const std::string& key /*!< The key to use to retrieve the model*/
                             );

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

  //! Run a model in the database
  void run_model(const std::string& key /*!< The key of the model to run*/,
                         std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                         std::vector<std::string> outputs /*!< The keys of the output tensors*/
                         );

  //! Run a script in the database
  void run_script(const std::string& key /*!< The key of the script to run*/,
                  const std::string& function /*!< The name of the function to run in the script*/,
                  std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                  std::vector<std::string> outputs /*!< The keys of the output tensors*/
                  );

  //! Check if a key exists
  bool key_exists(const std::string& key /*!< The key to check*/
                  );

  //! Poll database until key exists or number of tries is exceeded
  bool poll_key(const std::string& key /*!< The key to check*/,
                int poll_frequency_ms /*!< The frequency of polls*/,
                int num_tries /*!< The total number of tries*/);

  //! Set the data source (i.e. key prefix for get functions)
  void set_data_source(std::string source_id /*!< The prefix for retrieval commands*/
                       );

protected:
  //! Should this client treat tensors as fortran arrays
  bool _fortran_array;

  //! Array of database nodes
  std::vector<DBNode> _db_nodes;

  //! Number of database nodes
  unsigned _n_db_nodes;

  //! Retrieve environment variable SSDB
  std::string _get_ssdb();

  //! Populate hash slot and db node information
  void _populate_db_node_data(bool cluster);

  //! Parse the CommandReply for cluster slot information
  void _parse_reply_for_slots(CommandReply& reply);

  // Set a model using the provided string_view of the model
  inline void _set_model(const std::string& key /*!< The key to use to place the mdoel*/,
                         std::string_view model /*!< The model content*/,
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

  // Set a script using the provided string_view of the script
  inline void _set_script(const std::string& key /*!< The key to use to place the script*/,
                          const std::string& device /*!< The device to run the script*/,
                          std::string_view script /*!< The script content*/
                          );

  // Copy a tensor from src_key to dest_key
  inline void _copy_tensor(const std::string& src_key,
                           const std::string& dest_key);

  //! Execute a database command
  CommandReply _execute_command(Command& cmd /*!< The command to execute*/,
                                std::string prefix="" /*!< Prefix to specifically address cluster node*/
                                );

  void _execute_commands(CommandList& cmds /*!< The list of commands to execute*/,
                         std::string prefix="" /*!< The option server to process commands*/
                         );

  //! Parse tensor dimensions from a database reply
  std::vector<size_t> _get_tensor_dims(CommandReply& reply /*!< The database reply*/
                                    );

  //! Retrieve a string_view of data buffer string from the database reply
  std::string_view _get_tensor_data_blob(CommandReply& reply /*!< The database reply*/
                                         );

  //! Get tensor data type from the database reply
  TensorType _get_tensor_data_type(CommandReply& reply /*!< The database reply*/
                                    );

  //! Get a prefix for a model or script based on the hash slot
  std::string _get_crc16_prefix(uint64_t hash_slot /*!< The hash slot*/
                                );

  //! Perform an inverse CRC16 calculation
  uint64_t _crc16_inverse(uint64_t remainder /*!< The remainder of the CRC16 calculation*/
                          );

  //! Retrieve the optimum model prefix for the set of inputs
  DBNode* _get_model_script_db(const std::string& name,
                               std::vector<std::string>& inputs,
                               std::vector<std::string>& outputs);


  //! Determine if the key has a hash tag ("{" and "}") in it
  bool _has_hash_tag(const std::string& key);

  //! Get the hash slot of a key
  uint16_t _get_hash_slot(const std::string& key);

  //! Return the hash key set by the hash tag
  std::string _get_hash_tag(const std::string& key);

  //! Get DBNode (by index) that is responsible for the hash slot
  uint16_t _get_dbnode_index(uint16_t hash_slot,
                             unsigned lhs,
                             unsigned rhs);

  //! Set the prefixes for put and get for C++ client.
  void _set_prefixes_from_env();

  //! Return the put prefix
  inline std::string _put_prefix();

  //! Return the get prefix
  inline std::string _get_prefix();

  //! Run a model in the database that uses dagrun instead of modelrun
  void __run_model_dagrun(const std::string& key /*!< The key of the model to run*/,
                          std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                          std::vector<std::string> outputs /*!< The keys of the output tensors*/
                          );

  //! Append vector of keys with get prefixes
  void _append_with_get_prefix(std::vector<std::string>& keys);

  //! Append vector of keys with put prefixes
  void _append_with_put_prefix(std::vector<std::string>& keys);

  //! Gets a mapping between original names and temporary names for identical hash slot requirement
  std::vector<std::string>  _get_tmp_names(std::vector<std::string> names,
                                           std::string db_prefix);

  //! Copy a list of tensor names
  void _copy_tensors(std::vector<std::string> src,
                     std::vector<std::string> dest);

  //! Delete a list of keys
  void _delete_keys(std::vector<std::string> key,
                    std::string db_prefix="");

private:
  sw::redis::RedisCluster* redis_cluster;
  sw::redis::Redis* redis;

  //! MemoryList to handle get_model commands to return model string
  MemoryList<char> _model_queries;
  //! MemoryList to handle dimensions that are returned to the user
  MemoryList<size_t> _dim_queries;

  //! //! The _tensor_pack memory is not for querying by name, but is used
  //! to manage memory.
  TensorPack _tensor_memory;
  std::string _put_key_prefix;
  std::string _get_key_prefix;
  std::vector<std::string> _get_key_prefixes;
};

} //namespace SILC

#endif //SMARTSIM_CPP_CLIENT_H
