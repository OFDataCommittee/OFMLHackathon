#define PY_CLIENT_H
#ifdef __cplusplus

#include "client.h"
#include "pydataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

///@file

using namespace SILC;

namespace py = pybind11;

class PyClient;

/*!
*   \brief The PyClient class is a wrapper around the
           C++ client that is needed for the Python
           client.
*/
class PyClient
{
    public:

        /*!
        *   \brief PyClient constructor
        *   \param cluster Flag to indicate if a database cluster
        *                  is being used
        */
        PyClient(bool cluster);

        /*!
        *   \brief PyClient destructor
        */
        ~PyClient();

        /*!
        *   \brief Put a tensor into the database
        *   \param key The key to associate with this tensor
        *              in the database
        *   \param type The data type of the tensor
        *   \param data Numpy array with Pybind*
        */
        void put_tensor(std::string& key,
                        std::string& type,
                        py::array data);

        /*!
        *   \brief  Retrieve a tensor from the database.
        *   \details The memory of the data pointer used
        *            to construct the Numpy array is valid
        *            until the PyClient is destroyed.
        *            However, given that the memory
        *            associated with the return data is
        *            valid until PyClient destruction, this method
        *            should not be used repeatedly for large tensor
        *            data.  Instead it is recommended that the user
        *            use PyClient.unpack_tensor() for large tensor
        *            data and to limit memory use by the PyClient.
        *   \param key The name used to reference the tensor
        */
        py::array get_tensor(std::string& key);


        /*!
        *   \brief Send a PyDataSet object to the database
        *   \param dataset The PyDataSet object to send to the database
        */
        void put_dataset(PyDataset& dataset);


        /*!
        *   \brief Get a PyDataSet object from the database
        *   \param name The name of the dataset to retrieve
        *   \returns Pointer to the PyDataSet
        *            object retrieved from the database
        */
        PyDataset* get_dataset(const std::string& name);


        /*!
        *   \brief Set a script from file in the
        *          database for future execution
        *   \param key The key to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script_file The source file for the script
        */
        void set_script_from_file(const std::string& key,
                                const std::string& device,
                                const std::string& script_file);

        /*!
        *   \brief Set a script from std::string_view buffer in the
        *          database for future execution
        *   \param key The key to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script The script source in a std::string_view
        */
        void set_script(const std::string& key,
                        const std::string& device,
                        const std::string_view& script);

        /*!
        *   \brief Retrieve the script from the database
        *   \param key The key associated with the script
        *   \returns A std::string_view containing the script.
        *            The memory associated with the script
        *            is managed by the PyClient and is valid
        *            until the destruction of the PyClient.
        */
        std::string_view get_script(const std::string& key);

        /*!
        *   \brief Run a script function in the database using the
        *          specificed input and output tensors
        *   \param key The key associated with the script
        *   \param function The name of the function in the script
        *                   to run
        *   \param inputs The keys of inputs tensors to use
        *                 in the script
        *   \param outputs The keys of output tensors that
        *                 will be used to save script results
        */
        void run_script(const std::string& key,
                        const std::string& function,
                        std::vector<std::string>& inputs,
                        std::vector<std::string>& outputs);

        /*!
        *   \brief Set a model from std::string_view buffer in the
        *          database for future execution
        *   \param key The key to associate with the model
        *   \param model The model as a continuous buffer string_view
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model
        *                         execution
        *   \param tag A tag to attach to the model for
        *              information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        */
        void set_model(const std::string& key,
                        const std::string_view& model,
                        const std::string& backend,
                        const std::string& device,
                        int batch_size = 0,
                        int min_batch_size = 0,
                        const std::string& tag = "",
                        const std::vector<std::string>& inputs
                        = std::vector<std::string>(),
                        const std::vector<std::string>& outputs
                        = std::vector<std::string>());

        /*!
        *   \brief Set a model from file in the
        *          database for future execution
        *   \param key The key to associate with the model
        *   \param model_file The source file for the model
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model
        *                         execution
        *   \param tag A tag to attach to the model for
        *              information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        */
        void set_model_from_file(const std::string& key,
                                const std::string& model_file,
                                const std::string& backend,
                                const std::string& device,
                                int batch_size = 0,
                                int min_batch_size = 0,
                                const std::string& tag = "",
                                const std::vector<std::string>& inputs
                                    = std::vector<std::string>(),
                                const std::vector<std::string>& outputs
                                    = std::vector<std::string>());

        /*!
        *   \brief Run a model in the database using the
        *          specificed input and output tensors
        *   \param key The key associated with the model
        *   \param inputs The keys of inputs tensors to use
        *                 in the model
        *   \param outputs The keys of output tensors that
        *                 will be used to save model results
        */
        void run_model(const std::string& key,
                        std::vector<std::string> inputs,
                        std::vector<std::string> outputs);

        /*!
        *   \brief Retrieve the model from the database
        *   \param key The key associated with the model
        *   \returns A py:bytes object containing the model
        */
        py::bytes get_model(const std::string& key);

        /*!
        *   \brief Check if the key exists in the database at a
        *          specified frequency for a specified number
        *          of times
        *   \param key The key that will be checked in the database
        *   \param use_prefix Whether the key should be prefixed with
        *                     the client's data source prefix.
        *   \param poll_frequency_ms The frequency of checks for the
        *                            key in milliseconds
        *   \param num_tries The total number of times to check for
        *                    the specified number of keys.  If the
        *                    value is set to -1, the key will be
        *                    polled indefinitely.
        *   \returns Returns true if the key is found within the
        *            specified number of tries, otherwise false.
        */
        bool poll_key(const std::string& key,
                      bool use_prefix,
                      int poll_frequency_ms,
                      int num_tries);

        /*!
        *   \brief Check if the key exists in the database
        *   \param key The key that will be checked in the database
        *   \param use_prefix Whether the key should be prefixed with
        *                     the client's data source prefix.
        *   \returns Returns true if the key exists in the database
        */
        bool key_exists(const std::string& key, bool use_prefix);

        /*!
        *   \brief Set the data source (i.e. key prefix for
        *          get functions)
        *   \param source_id The prefix for retrieval commands
        */
        void set_data_source(const std::string& source_id);

        /*!
         * \brief Set whether tensor keys should be prefixed
         *        (e.g. in an ensemble). Prefixes will only be
         *        used if they were previously set through
         *        environment variables SSKEYIN and SSKEYOUT.
         *        By default, the client prefixes tensor keys
         *        when a prefix is available.
         *
         * \param use_prefix If set to true, all future operations
         *                   on tensors will use a prefix, if 
         *                   available.
         */
        void use_tensor_ensemble_prefix(bool use_prefix);
        
       
        /*!
         * \brief Set whether model and script keys should be prefixed
         *        (e.g. in an ensemble). Prefixes will only be
         *        used if they were previously set through
         *        environment variables SSKEYIN and SSKEYOUT.
         *        By default, the client does not prefix model keys.
         *
         * \param use_prefix If set to true, all future operations
         *                   on models will use a prefix, if 
         *                   available.
         */
        void use_model_ensemble_prefix(bool use_prefix);


        /*!
         * \brief Set whether dataset keys should be prefixed
         *        (e.g. in an ensemble). Prefixes will only be
         *        used if they were previously set through
         *        environment variables SSKEYIN and SSKEYOUT.
         *        By default, the client does not prefix dataset keys.
         *
         * \param use_prefix If set to true, all future operations
         *                   on datasets will use a prefix, if 
         *                   available.
         */
        void use_dataset_ensemble_prefix(bool use_prefix);

    private:

        /*!
        *   \brief Pointer to a Client object for
        *          executing server commands
        */
        Client* _client;

};
#endif //PY_CLIENT_H
