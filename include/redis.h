#ifndef SMARTSIM_CPP_REDIS_H
#define SMARTSIM_CPP_REDIS_H

#include "redisserver.h"

namespace SILC {

///@file
///\brief A class to manage a cluster connection and assist with cluster commands.

class Redis;

class Redis : public RedisServer
{
    public:
        //! Redis constructor.  Initializes default values but does not connect.
        Redis();

        //! Redis constructor.  Uses address provided to constructor.
        Redis(std::string address_port /*!< The address and port string*/
                     );

        //! Redis copy constructor
        Redis(const Redis& cluster) = delete;

        //! Redis copy assignment operator
        Redis& operator=(const Redis& cluster) = delete;

        //! Client destructor
        ~Redis();

        //! Run a Command on the Redis and return the CommandReply.
        virtual CommandReply run(Command& cmd /*!< The Command to run*/
                                 );

        //! Run a CommandList and return the last CommandReply.
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

        //! Set a script using the provided string_view of the script
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
        sw::redis::Redis* _redis;

        //! Connect to Redis cluster using a single address
        inline void _connect(std::string address_port /*!< A string formatted as tcp:://address:port for redis connection*/
                             );
};

} //namespace SILC

#endif //SMARTSIM_CPP_REDIS_H