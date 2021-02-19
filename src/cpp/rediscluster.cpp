#include "rediscluster.h"

using namespace SILC;

RedisCluster::RedisCluster() : RedisServer() {
    std::string address_port = this->_get_ssdb();
    this->_connect(address_port);
    this->_map_cluster();
    return;
}

RedisCluster::RedisCluster(std::string address_port) :
    RedisServer()
{
    this->_connect(address_port);
    this->_map_cluster();
    return;
}

RedisCluster::~RedisCluster()
{
    if(this->_redis_cluster)
        delete this->_redis_cluster;
}

CommandReply RedisCluster::run(Command& cmd)
{
    std::string db_prefix = this->_get_db_node_prefix(cmd);
    std::string_view sv_prefix(db_prefix.data(), db_prefix.size());
    sw::redis::Redis db = this->_redis_cluster->redis(sv_prefix, false);

    return this->_run(&db, cmd);
}

CommandReply RedisCluster::run(CommandList& cmds)
{
    CommandList::iterator cmd = cmds.begin();
    CommandList::iterator cmd_end = cmds.end();
    CommandReply reply;
    while(cmd != cmd_end) {
        this->run(**cmd);
        cmd++;
    }
    return reply;
}

bool RedisCluster::key_exists(const std::string& key)
{
    Command cmd;
    cmd.add_field("EXISTS");
    cmd.add_field(key, true);
    CommandReply reply = this->run(cmd);
    return reply.integer();
}

CommandReply RedisCluster::put_tensor(TensorBase& tensor)
{
    Command cmd;
    cmd.add_field("AI.TENSORSET");
    cmd.add_field(tensor.name(), true);
    cmd.add_field(tensor.type_str());
    cmd.add_fields(tensor.dims());
    cmd.add_field("BLOB");
    cmd.add_field_ptr(tensor.buf());
    return this->run(cmd);
}

CommandReply RedisCluster::get_tensor(const std::string& key)
{
    Command cmd;
    cmd.add_field("AI.TENSORGET");
    cmd.add_field(key, true);
    cmd.add_field("META");
    cmd.add_field("BLOB");
    return this->run(cmd);
}

CommandReply RedisCluster::rename_tensor(const std::string& key,
                                         const std::string& new_key)
{
    CommandReply reply;

    uint16_t key_hash_slot = this->_get_hash_slot(key);
    uint16_t new_key_hash_slot = this->_get_hash_slot(new_key);

    if(key_hash_slot == new_key_hash_slot) {
        Command cmd;
        cmd.add_field("RENAME");
        cmd.add_field(key, true);
        cmd.add_field(new_key, true);
        reply = this->run(cmd);
    }
    else {
        this->copy_tensor(key, new_key);
        reply = this->delete_tensor(key);
    }
    return reply;
}

CommandReply RedisCluster::delete_tensor(const std::string& key)
{
    Command cmd;
    cmd.add_field("UNLINK");
    cmd.add_field(key, true);
    return this->run(cmd);
}

CommandReply RedisCluster::copy_tensor(const std::string& src_key,
                                       const std::string& dest_key)
{
    //TODO can we do COPY for same hash slot or database (only for redis 6.2)?
    CommandReply cmd_get_reply;
    Command cmd_get;

    cmd_get.add_field("AI.TENSORGET");
    cmd_get.add_field(src_key, true);
    cmd_get.add_field("META");
    cmd_get.add_field("BLOB");
    cmd_get_reply = this->run(cmd_get);

    std::vector<size_t> dims =
        CommandReplyParser::get_tensor_dims(cmd_get_reply);
    std::string_view blob =
        CommandReplyParser::get_tensor_data_blob(cmd_get_reply);
    TensorType type =
        CommandReplyParser::get_tensor_data_type(cmd_get_reply);

    CommandReply cmd_put_reply;
    Command cmd_put;

    cmd_put.add_field("AI.TENSORSET");
    cmd_put.add_field(dest_key, true);
    cmd_put.add_field(TENSOR_STR_MAP.at(type));
    cmd_put.add_fields(dims);
    cmd_put.add_field("BLOB");
    cmd_put.add_field_ptr(blob);
    cmd_put_reply = this->run(cmd_put);

    return cmd_put_reply;
}

CommandReply RedisCluster::copy_tensors(const std::vector<std::string>& src,
                                        const std::vector<std::string>& dest)
{
    std::vector<std::string>::const_iterator src_it = src.cbegin();
    std::vector<std::string>::const_iterator src_it_end = src.cend();

    std::vector<std::string>::const_iterator dest_it = dest.cbegin();
    std::vector<std::string>::const_iterator dest_it_end = dest.cend();

    CommandReply reply;

    while(src_it!=src_it_end && dest_it!=dest_it_end)
    {
        reply = this->copy_tensor(*src_it, *dest_it);
        src_it++;
        dest_it++;
    }
    return reply;
}

CommandReply RedisCluster::set_model(const std::string& model_name,
                                     std::string_view model,
                                     const std::string& backend,
                                     const std::string& device,
                                     int batch_size,
                                     int min_batch_size,
                                     const std::string& tag,
                                     const std::vector<std::string>& inputs,
                                     const std::vector<std::string>& outputs
                                     )
{
    std::string prefixed_key;
    std::vector<DBNode>::const_iterator node =
      this->_db_nodes.cbegin();
    std::vector<DBNode>::const_iterator end_node =
      this->_db_nodes.cend();

    CommandReply reply;

    while(node!=end_node)
    {
        prefixed_key = "{" + node->prefix +
                       "}." + model_name;
        Command cmd;
        cmd.add_field("AI.MODELSET");
        cmd.add_field(prefixed_key, true);
        cmd.add_field(backend);
        cmd.add_field(device);
        if(tag.size()>0) {
            cmd.add_field("TAG");
            cmd.add_field(tag);
        }
        if(batch_size>0) {
            cmd.add_field("BATCHSIZE");
            cmd.add_field(std::to_string(batch_size));
        }
        if(min_batch_size>0) {
            cmd.add_field("MINBATCHSIZE");
            cmd.add_field(std::to_string(min_batch_size));
        }
        if(inputs.size()>0) {
            cmd.add_field("INPUTS");
            cmd.add_fields(inputs);
        }
        if(outputs.size()>0) {
            cmd.add_field("OUTPUTS");
            cmd.add_fields(outputs);
        }
        cmd.add_field("BLOB");
        cmd.add_field_ptr(model);
        reply = this->run(cmd);
        node++;
    }
    return reply;
}

CommandReply RedisCluster::set_script(const std::string& key,
                                      const std::string& device,
                                      std::string_view script)
{
    std::string prefixed_key;
    CommandReply reply;

    std::vector<DBNode>::const_iterator node =
        this->_db_nodes.cbegin();
    std::vector<DBNode>::const_iterator end_node =
        this->_db_nodes.cend();
    while(node!=end_node) {
        prefixed_key = "{" + node->prefix +
                        "}." + std::string(key);

        Command cmd;
        cmd.add_field("AI.SCRIPTSET");
        cmd.add_field(prefixed_key, true);
        cmd.add_field(device);
        cmd.add_field("SOURCE");
        cmd.add_field_ptr(script);
        reply = this->run(cmd);

        node++;
    }

    return reply;
}

CommandReply RedisCluster::run_model(const std::string& key,
                                     std::vector<std::string> inputs,
                                     std::vector<std::string> outputs)
{
    /*  For this version of run model, we have to copy all
        input and output tensors, so we will randomly select
        a model.  We can't use rand, because MPI would then
        have the same random number across all ranks.  Instead
        We will choose it based on the db of the first input tensor.
    */

    uint16_t hash_slot = this->_get_hash_slot(inputs[0]);
    uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                                this->_db_nodes.size()-1);
    DBNode* db = &(this->_db_nodes[db_index]);

    //Generate temporary names so that all keys go to same slot
    std::vector<std::string> tmp_inputs =
        _get_tmp_names(inputs, db->prefix);
    std::vector<std::string> tmp_outputs =
        _get_tmp_names(outputs, db->prefix);

    //Copy all input tensors to temporary names to align hash slots
    this->copy_tensors(inputs, tmp_inputs);

    std::string model_name = "{" + db->prefix +
                            "}." + std::string(key);

    Command cmd;
    CommandReply reply;
    cmd.add_field("AI.MODELRUN");
    cmd.add_field(model_name, true);
    cmd.add_field("INPUTS");
    cmd.add_fields(tmp_inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(tmp_outputs);
    reply = this->run(cmd);

    this->copy_tensors(tmp_outputs, outputs);

    std::vector<std::string> keys_to_delete;
    keys_to_delete.insert(keys_to_delete.end(),
                            tmp_outputs.begin(),
                            tmp_outputs.end());
    keys_to_delete.insert(keys_to_delete.end(),
                            tmp_inputs.begin(),
                            tmp_inputs.end());

    this->_delete_keys(keys_to_delete);

    return reply;
}

CommandReply RedisCluster::run_script(const std::string& key,
                                      const std::string& function,
                                      std::vector<std::string> inputs,
                                      std::vector<std::string> outputs)
{
    uint16_t hash_slot = this->_get_hash_slot(inputs[0]);
    uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                                this->_db_nodes.size()-1);
    DBNode* db = &(this->_db_nodes[db_index]);

    //Generate temporary names so that all keys go to same slot
    std::vector<std::string> tmp_inputs =
        _get_tmp_names(inputs, db->prefix);
    std::vector<std::string> tmp_outputs =
        _get_tmp_names(outputs, db->prefix);

    //Copy all input tensors to temporary names to align hash slots
    this->copy_tensors(inputs, tmp_inputs);

    std::string script_name = "{" + db->prefix +
                            "}." + std::string(key);
    Command cmd;
    CommandReply reply;
    cmd.add_field("AI.SCRIPTRUN");
    cmd.add_field(script_name, true);
    cmd.add_field(function);
    cmd.add_field("INPUTS");
    cmd.add_fields(tmp_inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(tmp_outputs);
    reply = this->run(cmd);

    this->copy_tensors(tmp_outputs, outputs);

    std::vector<std::string> keys_to_delete;
    keys_to_delete.insert(keys_to_delete.end(),
                            tmp_outputs.begin(),
                            tmp_outputs.end());
    keys_to_delete.insert(keys_to_delete.end(),
                            tmp_inputs.begin(),
                            tmp_inputs.end());

    this->_delete_keys(keys_to_delete);
    return reply;
}

CommandReply RedisCluster::get_model(const std::string& key)
{
    std::string prefixed_str =
        "{" + this->_db_nodes[0].prefix +
        "}." + key;

    Command cmd;
    cmd.add_field("AI.MODELGET");
    cmd.add_field(prefixed_str, true);
    cmd.add_field("BLOB");
    return this->run(cmd);
}

CommandReply RedisCluster::get_script(const std::string& key)
{
    std::string prefixed_str =
        "{" + this->_db_nodes[0].prefix +
        "}." + std::string(key);

    Command cmd;
    cmd.add_field("AI.SCRIPTGET");
    cmd.add_field(prefixed_str, true);
    cmd.add_field("SOURCE");
    return this->run(cmd);
}


inline void RedisCluster::_connect(std::string address_port)
{
    int n_connection_trials = 10;

    while(n_connection_trials > 0) {
        try {
            this->_redis_cluster = new sw::redis::RedisCluster(address_port);
            n_connection_trials = -1;
        }
        catch (sw::redis::TimeoutError &e) {
          std::cout << "WARNING: Caught redis TimeoutError: "
                    << e.what() << std::endl;
          std::cout << "WARNING: TimeoutError occurred with "\
                       "initial client connection.";
          std::cout << "WARNING: "<< n_connection_trials
                      << " more trials will be made.";
          n_connection_trials--;
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    if(n_connection_trials==0)
        throw std::runtime_error("A connection could not be "\
                                 "established to the redis cluster.");
    return;
}

inline void RedisCluster::_map_cluster()
{
    this->_db_nodes.clear();

    Command cmd;
    cmd.add_field("CLUSTER");
    cmd.add_field("SLOTS");
    CommandReply reply(this->_redis_cluster->
                 command(cmd.begin(), cmd.end()));
    this->_parse_reply_for_slots(reply);
    return;
}

std::string RedisCluster::_get_db_node_prefix(Command& cmd)
{
    std::vector<std::string> keys = cmd.get_keys();

    if(keys.size()==0)
        throw std::runtime_error("Command " + cmd.to_string() +
                                 " does not have a key value.");

    std::vector<std::string>::iterator key_it =
        keys.begin();
    std::vector<std::string>::iterator key_it_end =
        keys.end();

    uint16_t hash_slot;
    uint16_t db_index;
    std::string prefix;

    while(key_it!=key_it_end) {
        hash_slot = this->_get_hash_slot(*key_it);
        db_index = this->_get_dbnode_index(hash_slot, 0,
                                           this->_db_nodes.size()-1);
        if(prefix.size()==0) {
            prefix = this->_db_nodes[db_index].prefix;
        }
        else{
            if(prefix!=this->_db_nodes[db_index].prefix) {
                throw std::runtime_error("Multi-key commands are "\
                                         "not valid: " +
                                         cmd.to_string());
            }
        }
        key_it++;
    }
    return prefix;
}

inline void RedisCluster::_parse_reply_for_slots(CommandReply& reply)
{
    /* This function parses a CommandReply for cluster slot
    information.
    Each reply element of the main message, of which there should
    be n_db_nodes, is:
    0) (integer) min slot
    1) (integer) max slot
    2) 0) "ip address"
       1) (integer) port
       2) "name"
    */
    size_t n_db_nodes = reply.n_elements();
    this->_db_nodes = std::vector<DBNode>(n_db_nodes);

    for(int i=0; i<n_db_nodes; i++) {
        this->_db_nodes[i].lower_hash_slot = reply[i][0].integer();
        this->_db_nodes[i].upper_hash_slot = reply[i][1].integer();
        this->_db_nodes[i].ip = std::string(reply[i][2][0].str(),
                                            reply[i][2][0].str_len());
        this->_db_nodes[i].port = reply[i][2][1].integer();
        this->_db_nodes[i].name = std::string(reply[i][2][2].str(),
                                              reply[i][2][2].str_len());
        bool acceptable_prefix = false;
        int n_hashes = this->_db_nodes[i].upper_hash_slot -
                       this->_db_nodes[i].lower_hash_slot + 1;
        int k = 0;
        while(!acceptable_prefix && k<=n_hashes) {
            this->_db_nodes[i].prefix = this->_get_crc16_prefix(
                                        this->_db_nodes[i].lower_hash_slot+k);
            std::string prefix = this->_db_nodes[i].prefix;
            bool found_bracket = false;
            for(int j=0; j<prefix.size(); j++) {
                if(prefix[j] == '}')
                found_bracket = true;
            }
            if(!found_bracket)
                acceptable_prefix=true;
            k++;
        }
        if(k>n_hashes)
            throw std::runtime_error("A prefix could not be generated "\
                                     "for this cluster config.");
    }
    //Put the vector of db nodes in order based on lower hash slot
    std::sort(this->_db_nodes.begin(), this->_db_nodes.end());
    return;
}

std::string RedisCluster::_get_crc16_prefix(uint64_t hash_slot)
{
    uint64_t byte_filter = 255;
    uint64_t crc_out = this->_crc16_inverse(hash_slot);
    crc_out = crc_out >> 16;
    //Get the two character prefix
    char* prefix = new char[2];
    for(int i=1; i>=0; i--) {
        prefix[i] = (crc_out&byte_filter);
        crc_out = crc_out>>8;
    }
    std::string prefix_str = std::string(prefix, 2);
    delete[] prefix;
    return prefix_str;
}

uint64_t RedisCluster::_crc16_inverse(uint64_t remainder)
{
    uint64_t digit = 1;
    uint64_t poly = 69665; //x^16 + x^12 + x^5 + 1

    for(int i=0; i<16; i++) {
        if(remainder&digit)
        remainder = remainder^poly;
        digit=digit<<1;
        poly=poly<<1;
    }
    return remainder;
}

bool RedisCluster::_has_hash_tag(const std::string& key)
{
    size_t first = key.find('{');
    size_t second = key.find('}');
    if(first == std::string::npos ||
        second == std::string::npos)
        return false;
    else if(second < first)
        return false;
    else
        return true;
}

std::string RedisCluster::_get_hash_tag(const std::string& key)
{
    size_t first = key.find('{');
    size_t second = key.find('}');
    if(first == std::string::npos ||
        second == std::string::npos)
        return key;
    else if(second < first)
        return key;
    else
        return key.substr(first+1,second-first-1);
}

uint16_t RedisCluster::_get_hash_slot(const std::string& key)
{
    std::string hash_key;
    if(this->_has_hash_tag(key))
        hash_key = this->_get_hash_tag(key);

    else
        hash_key = key;
    return sw::redis::crc16(hash_key.c_str(),
                            hash_key.size()) % 16384;
}

uint16_t RedisCluster::_get_dbnode_index(uint16_t hash_slot,
                                   unsigned lhs, unsigned rhs)
{
    uint16_t m = (lhs + rhs)/2;
    if(this->_db_nodes[m].lower_hash_slot<=hash_slot &&
        this->_db_nodes[m].upper_hash_slot>=hash_slot) {
        return m;
        }
    else {
        if(this->_db_nodes[m].lower_hash_slot > hash_slot)
        return this->_get_dbnode_index(hash_slot, lhs, m-1);
        else
        return this->_get_dbnode_index(hash_slot, m+1, rhs);
    }
}

std::vector<std::string>
RedisCluster::_get_tmp_names(std::vector<std::string> names,
                             std::string db_prefix)
{
    std::vector<std::string> tmp;
    std::vector<std::string>::iterator it = names.begin();
    std::vector<std::string>::iterator it_end = names.end();
    while(it!=it_end) {
        std::string new_key = "{" + db_prefix + "}." +
                            *it + ".TMP";
        tmp.push_back(new_key);
        it++;
    }
    return tmp;
}

void RedisCluster::_delete_keys(std::vector<std::string> keys)
{
    CommandReply reply;
    Command cmd;
    cmd.add_field("DEL");
    cmd.add_fields(keys, true);
    reply = this->run(cmd);
    return;
}

void RedisCluster::__run_model_dagrun(const std::string& key,
                                      std::vector<std::string> inputs,
                                      std::vector<std::string> outputs)
{
    /*This function will run a RedisAI model.  Because the RedisAI
    AI.RUNMODEL and AI.DAGRUN commands assume that the tensors
    and model are all on the same node.  As a result, we will
    have to retrieve all input tensors that are not on the same
    node as the model and set temporary
    */

    //TODO We need to make sure that no other clients are using the
    //same keys and model because we may end up overwriting or having
    //race conditions on who can use the model, etc.

    DBNode* db = this->_get_model_script_db(key, inputs, outputs);

    //Create list of input tensors that do not hash to db slots
    std::unordered_set<std::string> remote_inputs;
    for(int i=0; i<inputs.size(); i++) {
        uint16_t hash_slot = this->_get_hash_slot(inputs[i]);
        if(hash_slot < db->lower_hash_slot ||
        hash_slot > db->upper_hash_slot)
        remote_inputs.insert(inputs[i]);
    }

    //Retrieve tensors that do not hash to db,
    //rename the tensors to {prefix}.tensor_name.TMP
    //TODO we need to make sure users don't use the .TMP suffix
    //or check that the key does not exist
    for(int i=0; i<inputs.size(); i++) {
        if(remote_inputs.count(inputs[i])>0) {
        std::string new_key = "{" + db->prefix + "}." +
                                inputs[i] + ".TMP";
        this->copy_tensor(inputs[i], new_key);
        remote_inputs.erase(inputs[i]);
        remote_inputs.insert(new_key);
        inputs[i] = new_key;
        }
    }

    //Create a renaming scheme for output tensor
    std::unordered_map<std::string, std::string> remote_outputs;
    for(int i=0; i<outputs.size(); i++) {
        uint16_t hash_slot = this->_get_hash_slot(outputs[i]);
        if(hash_slot < db->lower_hash_slot ||
        hash_slot > db->upper_hash_slot) {
            std::string tmp_name = "{" + db->prefix + "}." +
                                outputs[i] + ".TMP";
            remote_outputs.insert({outputs[i], tmp_name});
            outputs[i] = remote_outputs[outputs[i]];
        }
    }

    std::string model_name = "{" + db->prefix +
                            "}." + std::string(key);
    Command cmd;

    cmd.add_field("AI.DAGRUN");
    cmd.add_field("LOAD");
    cmd.add_field(std::to_string(inputs.size()));
    cmd.add_fields(inputs);
    cmd.add_field("PERSIST");
    cmd.add_field(std::to_string(outputs.size()));
    cmd.add_fields(outputs);
    cmd.add_field("|>");
    cmd.add_field("AI.MODELRUN");
    cmd.add_field(model_name, true);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);
    this->run(cmd);

    //Delete temporary input tensors
    std::unordered_set<std::string>::const_iterator i_it
        = remote_inputs.begin();
    std::unordered_set<std::string>::const_iterator i_it_end
        = remote_inputs.end();
    while(i_it!=i_it_end) {
        this->delete_tensor(*i_it);
        i_it++;
    }

    //Move temporary output to the correct location and
    //delete temporary output tensors
    std::unordered_map<std::string, std::string>::const_iterator j_it
        = remote_outputs.begin();
    std::unordered_map<std::string, std::string>::const_iterator j_it_end
        = remote_outputs.end();
    while(j_it!=j_it_end) {
        this->rename_tensor(j_it->second, j_it->first);
        j_it++;
    }

    return;
}

DBNode* RedisCluster::_get_model_script_db(const std::string& name,
                                           std::vector<std::string>& inputs,
                                           std::vector<std::string>& outputs)
{
    /* This function calculates the optimal model name to use
    to run the provided inputs.  If a cluster is not being used,
    the model name is returned, else a prefixed model name is returned.
    */

    //TODO we should randomly choose the max if there are multiple
    //maxes

    std::vector<int> hash_slot_tally(this->_db_nodes.size(), 0);

    for(int i=0; i<inputs.size(); i++) {
        uint16_t hash_slot = this->_get_hash_slot(inputs[i]);
        uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                                    this->_db_nodes.size());
        hash_slot_tally[db_index]++;
    }

    for(int i=0; i<outputs.size(); i++) {
        uint16_t hash_slot = this->_get_hash_slot(outputs[i]);
        uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                                    this->_db_nodes.size());
        hash_slot_tally[db_index]++;
    }

    //Determine which DBNode has the most hashes
    int max_hash = -1;
    DBNode* db = 0;
    for(int i=0; i<this->_db_nodes.size(); i++) {
        if(hash_slot_tally[i] > max_hash) {
        max_hash = hash_slot_tally[i];
        db = &(this->_db_nodes[i]);
        }
    }
    return db;
}
