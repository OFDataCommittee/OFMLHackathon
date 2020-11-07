#include "dbnode.h"

DBNode::DBNode()
{
    /* Default DBNode constructor
    */
   this->name = "";
   this->ip = "";
   this->port = -1;
   this->lower_hash_slot = -1;
   this->upper_hash_slot = -1;
}

DBNode::DBNode(std::string ip, std::string name,
               uint64_t port, uint64_t l_slot,
               uint64_t u_slot, std::string prefix)
{
    /* Full constructor to create a DBNode
    */
    this->name = name;
    this->ip = ip;
    this->port = port;
    this->lower_hash_slot = l_slot;
    this->upper_hash_slot = u_slot;
    this->prefix = prefix;
}

DBNode::~DBNode()
{
}

bool DBNode::operator<(const DBNode& db_node) const
{
    return this->lower_hash_slot <
        db_node.lower_hash_slot;
}