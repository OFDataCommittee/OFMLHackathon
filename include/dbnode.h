#ifndef SMARTSIM_DBNODE_H
#define SMARTSIM_DBNODE_H

#include "stdlib.h"
#include <string>

class DBNode;

///@file
///\brief This class stores database node information
class DBNode{

    public:
        //! Default DBNode constructor
        DBNode();

        //! Full DBNode constructor
        DBNode(std::string name,
               std::string ip,
               uint64_t port,
               uint64_t l_slot,
               uint64_t u_slot,
               std::string prefix
               );

        //! Default DBNode destructor
        ~DBNode();

        //! Inequality comparator
        bool operator<(const DBNode& db_node) const;

        //! IP address of the database node
        std::string ip;

        //! Port of the database node
        uint64_t port;

        //! Name of the database node
        std::string name;

        //! Lower hash slot limit
        uint64_t lower_hash_slot;

        //! Upper hash slot limit
        uint64_t upper_hash_slot;

        //! CRC16 prefix
        std::string prefix;

};

#endif //SMARTSIM_DBNODE_H