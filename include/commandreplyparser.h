#ifndef SMARTSIM_CPP_COMMANDREPLYPARSER_H
#define SMARTSIM_CPP_COMMANDREPLYPARSER_H

#include <vector>
#include "commandreply.h"
#include "tensorbase.h"
#include "enums/cpp_tensor_type.h"

namespace SILC {

namespace CommandReplyParser {

inline std::vector<size_t> get_tensor_dims(CommandReply& reply)
{
    /* This function will fill a vector with the dimensions of the
    tensor.  We assume right now that the META reply is always
    in the same order and we can index base reply elements array;
    */

    if(reply.n_elements() < 6)
        throw std::runtime_error("The message does not have the "\
                                "correct number of fields");

    size_t n_dims = reply[3].n_elements();
    std::vector<size_t> dims(n_dims);

    for(size_t i=0; i<n_dims; i++) {
        dims[i] = reply[3][i].integer();
    }

    return dims;
}

inline std::string_view get_tensor_data_blob(CommandReply& reply)
{
    /* Returns a string view of the data tensor blob value
    */

    //We are going to assume right now that the meta data reply
    //is always in the same order and we are going to just
    //index into the base reply.

    if(reply.n_elements() < 6)
        throw std::runtime_error("The message does not have the "\
                                "correct number of fields");

    return std::string_view(reply[5].str(), reply[5].str_len());
}

inline TensorType get_tensor_data_type(CommandReply& reply)
{
    /* Returns a string of the tensor type
    */

    //We are going to assume right now that the meta data reply
    //is always in the same order and we are going to just
    //index into the base reply.

    if(reply.n_elements() < 2)
        throw std::runtime_error("The message does not have the correct "\
                                "number of fields");

    return TENSOR_TYPE_MAP.at(std::string(reply[1].str(),
                                            reply[1].str_len()));
}

} //namespace CommandReplyParser

} //namespace SILC

#endif //SMARTSIM_CPP_COMMANDREPLYPARSER_H
