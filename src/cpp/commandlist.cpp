#include "commandlist.h"

using namespace SILC;

CommandList::CommandList()
{
}

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
    /* Returns a iterator pointing to the first
    command
    */
    return this->_commands.begin();
}

CommandList::const_iterator CommandList::cbegin()
{
    /* Returns a const_iterator pointing to the first
    command
    */
    return this->_commands.cbegin();
}

CommandList::iterator CommandList::end()
{
    /* Returns a iterator pointing to the past-the-end
    command
    */
    return this->_commands.end();
}

CommandList::const_iterator CommandList::cend()
{
    /* Returns a const_iterator pointing to the past-the-end
    command
    */
    return this->_commands.cend();
}