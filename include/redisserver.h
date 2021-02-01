#ifndef SMARTSIM_CPP_REDISSERVER_H
#define SMARTSIM_CPP_REDISSERVER_H

#include <thread>
#include <iostream>
#include <sw/redis++/redis++.h>
#include "command.h"
#include "commandreply.h"
#include "commandreplyparser.h"
#include "commandlist.h"
#include "tensorbase.h"

///@file
///\brief Abstract class that defines interface for objects that execute commands on server.

namespace SILC {

class RedisServer;

class RedisServer {

    public:

        //! Default constructor
        RedisServer() = default;

        //! Destructor
        virtual ~RedisServer() = default;

        //! Run a single-key command on the server
        virtual CommandReply run(Command& cmd) = 0;

        //! Run a CommandList and return the last CommandReply.
        virtual CommandReply run(CommandList& /*!< The CommandList to run*/
                                 ) = 0;

        //! Check if a key exists
        virtual bool key_exists(const std::string& key /*!< The key to check*/
                                ) = 0;

        //! Put the RedisAI tensor on the server
        virtual CommandReply put_tensor(TensorBase& tensor /*!< The Tensor to put*/
                                        ) = 0;

        //! Get the RedisAI tensor on the server
        virtual CommandReply get_tensor(const std::string& key /*!< The key of the Tensor*/
                                        ) = 0;

        //! Rename a tensor
        virtual CommandReply rename_tensor(const std::string& key /*!< The original tensor key*/,
                                           const std::string& new_key /*!< The new tensor key*/
                                           ) = 0;

        //! Delete a tensor
        virtual CommandReply delete_tensor(const std::string& key /*!< The key of tensor to delete*/
                                           ) = 0;

        //! This method will copy a tensor to the destination key
        virtual CommandReply copy_tensor(const std::string& src_key /*!< The source tensor key*/,
                                         const std::string& dest_key /*!< The destination tensor key*/
                                         ) = 0;

        //! Copy a vector of tensors
        virtual CommandReply copy_tensors(const std::vector<std::string>& src /*!< The source tensor keys*/,
                                          const std::vector<std::string>& dest /*!< The destination tensor keys*/
                                          ) = 0;

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
                                       ) = 0;

        // Set a script using the provided string_view of the script
        virtual CommandReply set_script(const std::string& key /*!< The key to use to place the script*/,
                                        const std::string& device /*!< The device to run the script*/,
                                        std::string_view script /*!< The script content*/
                                        ) = 0;

        //! Run a model in the database
        virtual CommandReply run_model(const std::string& key /*!< The key of the model to run*/,
                                       std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                                       std::vector<std::string> outputs /*!< The keys of the output tensors*/
                                       ) = 0;

        //! Run a script in the database
        virtual CommandReply run_script(const std::string& key /*!< The key of the script to run*/,
                                        const std::string& function /*!< The name of the function to run in the script*/,
                                        std::vector<std::string> inputs /*!< The keys of the input tensors*/,
                                        std::vector<std::string> outputs /*!< The keys of the output tensors*/
                                        ) = 0;

        //! Get a model in the database
        virtual CommandReply get_model(const std::string& key /*!< The key to use to retrieve the model*/
                                       ) = 0;

        //! Get the script from the database
        virtual CommandReply get_script(const std::string& key /*!< The key to use to retrieve the script*/
                                        ) = 0;

    protected:

        //! Run a Command using a provided sw:redis::Redis object pointer
        CommandReply _run(sw::redis::Redis* redis, Command& cmd);

        //! Retrieve environment variable SSDB
        std::string _get_ssdb();

        //! Check that the SSDB environment string conforms to permissable characters
        void _check_ssdb_string(const std::string& env_str);

};

} // namespace SILC

#endif //SMARTSIM_CPP_REDISSERVER_H
