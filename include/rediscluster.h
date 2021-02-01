#ifndef SMARTSIM_CPP_CLUSTER_H
#define SMARTSIM_CPP_CLUSTER_H

#include <unordered_set>
#include "redisserver.h"
#include "dbnode.h"

namespace SILC {

///@file
///\brief A class to manage a cluster connection and assist with cluster commands.

class RedisCluster;

class RedisCluster : public RedisServer
{
    public:
        //! RedisCluster constructor.  Initializes default values but does not connect.
        RedisCluster();

        //! RedisCluster constructor.  Uses address provided to constructor.
        RedisCluster(std::string address_port /*!< The address and port string*/
                     );

        //! RedisCluster copy constructor
        RedisCluster(const RedisCluster& cluster) = delete;

        //! RedisCluster copy assignment operator
        RedisCluster& operator=(const RedisCluster& cluster) = delete;

        //! Client destructor
        ~RedisCluster();

        //! Run a Command on the RedisCluster and return the CommandReply.
        virtual CommandReply run(Command& cmd /*!< The Command to run*/
                                 );

        //! Run a CommandList on the RedisCluster and return the last CommandReply.
        virtual CommandReply run(CommandList& /*!< The CommandList to run*/
                                 );

        //! Check if a key exists
        virtual bool key_exists(const std::string& key /*!< The key to check*/
                                );

        //! Put the RedisAI tensor on the server
        virtual CommandReply put_tensor(TensorBase& tensor /*!< The Tensor to put*/
                                        );

        //! Get the RedisAI tensor on the server
        virtual CommandReply get_tensor(const std::string& key /*!< The key of the Tensor*/
                                        );

        //! Rename a tensor
        virtual CommandReply rename_tensor(const std::string& key /*!< The original tensor key*/,
                                           const std::string& new_key /*!< The new tensor key*/
                                           );

        //! Delete a tensor
        virtual CommandReply delete_tensor(const std::string& key /*!< The key of tensor to delete*/
                                           );

        //! This method will copy a tensor to the destination key
        virtual CommandReply copy_tensor(const std::string& src_key /*!< The source tensor key*/,
                                         const std::string& dest_key /*!< The destination tensor key*/
                                         );

        //! Copy a vector of tensors
        virtual CommandReply copy_tensors(const std::vector<std::string>& src /*!< The source tensor keys*/,
                                          const std::vector<std::string>& dest /*!< The destination tensor keys*/
                                          );

        //! Set a model on the server
        virtual CommandReply set_model(const std::string& key /*!< The key to use to place the model*/,
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
        virtual CommandReply set_script(const std::string& key /*!< The key to use to place the script*/,
                                        const std::string& device /*!< The device to run the script*/,
                                        std::string_view script /*!< The script content*/
                                        );

        //! Run a model in the database
        virtual CommandReply run_model(const std::string& key /*!< The key of the model to run*/,
                                       std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                                       std::vector<std::string> outputs /*!< The keys of the output tensors*/
                                       );

        //! Run a script in the database
        virtual CommandReply run_script(const std::string& key /*!< The key of the script to run*/,
                                        const std::string& function /*!< The name of the function to run in the script*/,
                                        std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                                        std::vector<std::string> outputs /*!< The keys of the output tensors*/
                                        );

        //! Get a model in the database
        virtual CommandReply get_model(const std::string& key /*!< The key to use to retrieve the model*/
                                       );

        //! Get the script from the database
        virtual CommandReply get_script(const std::string& key /*!< The key to use to retrieve the script*/
                                        );

    private:

        //! Redis cluster object pointer
        sw::redis::RedisCluster* _redis_cluster;

        //! Vector of database nodes that make up the cluster
        std::vector<DBNode> _db_nodes;

        //! Connect to Redis cluster using a single address
        inline void _connect(std::string address_port /*!< A string formatted as tcp:://address:port for redis connection*/
                             );

        //! Map the RedisCluster via the CLUSTER SLOTS command
        inline void _map_cluster();

        //! Get the prefix that can be used to address the correct database for a given command
        std::string _get_db_node_prefix(Command& cmd /*!< The Command to analyze for database prefix*/
                                        );

        //! This command parses the CommandReply for the DBNode information
        inline void _parse_reply_for_slots(CommandReply& reply /*!< The CLUSTER INFO CommandReply*/
                                           );

        //! Get a DBNode prefix for the provided hash slot
        std::string _get_crc16_prefix(uint64_t hash_slot /*!< The hash slot*/
                                      );

        //! Perform an inverse CRC16 calculation
        uint64_t _crc16_inverse(uint64_t remainder /*!< The remainder of the CRC16 calculation*/
                                );

        //! Determine if the key has a hash tag ("{" and "}") in it
        bool _has_hash_tag(const std::string& key /*!< The key to inspect for hash tag*/
                           );

        //! Return the hash key set by the hash tag
        std::string _get_hash_tag(const std::string& key /*!< The key containing a hash tag*/
                                  );

        //! Get the hash slot of a key
        uint16_t _get_hash_slot(const std::string& key /*!< Get the key to process into hash slot*/)
                                ;

        //! Get DBNode (by index) that is responsible for the hash slot
        uint16_t _get_dbnode_index(uint16_t hash_slot /*!< The hash slot to search for*/,
                                   unsigned lhs /*!< Left search boundary DBNode*/,
                                   unsigned rhs /*!< Right search bondary DBNode*/
                                   );

        //! Gets a mapping between original names and temporary names for identical hash slot requirement
        std::vector<std::string>  _get_tmp_names(std::vector<std::string> names,
                                                 std::string db_prefix);

        //! Delete a list of keys
        void _delete_keys(std::vector<std::string> key
                          );

        //! Run a model in the database that uses dagrun instead of modelrun
        void __run_model_dagrun(const std::string& key /*!< The key of the model to run*/,
                                std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                                std::vector<std::string> outputs /*!< The keys of the output tensors*/
                                );

        //! Retrieve the optimum model prefix for the set of inputs
        DBNode* _get_model_script_db(const std::string& name,
                                     std::vector<std::string>& inputs,
                                     std::vector<std::string>& outputs);


};

} //namespace SILC

#endif //SMARTSIM_CPP_CLUSTER_H