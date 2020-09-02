#include "dataset.h"

DataSet::DataSet(const char* name)
{
    this->name = std::string(name);
}

DataSet::~DataSet() {}

void DataSet::add_tensor(const char* name, const char* type, void* data, std::vector<int> dims)
{
    tensors.push_back(Tensor(name, type, data, dims));
}

std::vector<Command> DataSet::_create_send_commands()
{
    //TODO need to add the prefix and suffix to the tensorget keys
    std::vector<Command> commands;

    std::vector<Tensor>::iterator it = this->tensors.begin();
    while(it != this->tensors.end()) {
        commands.push_back(it->generate_send_command("",""));
        it++;
    }

    return commands;
}