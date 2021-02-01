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
#include "redisserver.h"
#include "rediscluster.h"
#include "redis.h"
#include <fstream>
#include "dataset.h"
#include "sharedmemorylist.h"
#include "command.h"
#include "commandlist.h"
#include "commandreply.h"
#include "commandreplyparser.h"
#include "tensorbase.h"
#include "tensor.h"
#include "dbnode.h"
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
    Client(bool cluster /*!< Flag to indicate if a database cluster is being used*/,
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
                        const std::string& new_key /*!< The new tensor key*/
                        );

    //! Delete a tensor
    void delete_tensor(const std::string& key /*!< The key of tensor to delete*/
                        );

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
                    int num_tries /*!< The total number of tries*/
                    );

    //! Set the data source (i.e. key prefix for get functions)
    void set_data_source(std::string source_id /*!< The prefix for retrieval commands*/
                         );

protected:
    //! Should this client treat tensors as fortran arrays
    bool _fortran_array;

    //! Abstract base class used to generalized running with cluster or non-cluster
    RedisServer* _redis_server;

    //! Pointer to RedisCluster object for redis cluster executions
    RedisCluster* _redis_cluster;

    //! Pointer to Redis object for non-cluster executions
    Redis* _redis;

    //! Set the prefixes for put and get for C++ client.
    void _set_prefixes_from_env();

    //! Return the put prefix
    inline std::string _put_prefix();

    //! Return the get prefix
    inline std::string _get_prefix();

    //! Append vector of keys with get prefixes
    inline void _append_with_get_prefix(std::vector<std::string>& keys);

    //! Append vector of keys with put prefixes
    inline void _append_with_put_prefix(std::vector<std::string>& keys);

private:

    //! SharedMemoryList to handle get_model commands to return model string
    SharedMemoryList<char> _model_queries;
    //! SharedMemoryList to handle dimensions that are returned to the user
    SharedMemoryList<size_t> _dim_queries;

    //! The _tensor_pack memory is not for querying by name, but is used
    //! to manage memory.
    TensorPack _tensor_memory;
    std::string _put_key_prefix;
    std::string _get_key_prefix;
    std::vector<std::string> _get_key_prefixes;
};

} //namespace SILC

#endif //SMARTSIM_CPP_CLIENT_H
