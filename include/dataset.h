#ifndef SMARTSIM_DATASET_H
#define SMARTSIM_DATASET_H

#include "stdlib.h"
#include <string>
#include <vector>
#include "tensor.h"
#include "command.h"

///@file
///\brief The DataSet class encapsulating numeric data and metadata.

class DataSet;

class DataSet
{
    public:

        //! Dataset constructor
        DataSet(const char* name /*!< The name used to reference the dataset*/
        );
        //! Dataset destructor
        ~DataSet();
        //! Add a tensor to the dataset
        void add_tensor(const char* name, const char* type, void* data, std::vector<int> dims);
        std::string name;
        std::vector<Tensor> tensors;
    private:
        std::vector<Command> _create_send_commands();
};

#endif //SMARTSIM_DATASET_H