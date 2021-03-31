#include "commandlist.h"

using namespace SmartRedis;

CommandList::~CommandList()
{
    std::vector<Command*>::iterator it = this->_commands.begin();
    std::vector<Command*>::iterator it_end = this->_commands.end();
    while(it != it_end) {
        delete (*it);
        it++;
    }
}

Command* CommandList::add_command()
{
    this->_commands.push_back(new Command());
    return this->_commands.back();
}

CommandList::iterator CommandList::begin()
{
    return this->_commands.begin();
}

CommandList::const_iterator CommandList::cbegin()
{
    return this->_commands.cbegin();
}

CommandList::iterator CommandList::end()
{
    return this->_commands.end();
}

CommandList::const_iterator CommandList::cend()
{
    return this->_commands.cend();
}