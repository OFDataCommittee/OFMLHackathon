#include <iostream>
#include "crc16.h"

uint64_t crc16_inverse(uint64_t remainder)
{
    uint64_t digit = 1;
    uint64_t poly = 69665; //x^16 + x^12 + x^5 + 1

    for(int i=0; i<16; i++) {
        //std::cout<<"i = "<<i<<std::endl;
        //std::cout<<"remainder:"<<std::endl;
        //std::cout << std::bitset<64>(remainder)<<std::endl;
        //std::cout<<"poly:"<<std::endl;
        //std::cout << std::bitset<64>(poly)<<std::endl;
        if(remainder&digit) {
            remainder = remainder^poly;
        }
        digit=digit<<1;
        poly=poly<<1;
    }
    //std::cout<<"Answer: "<<std::endl;
    //std::cout << std::bitset<64>(remainder)<<std::endl;
    return remainder;
}

char* get_prefix(uint64_t prefix_int)
{
    /* This takes the raw 64bit output from crc16_inverse
    and returns a 2 character prefix
    */
    uint64_t byte_filter = 255;
    prefix_int = prefix_int >> 16;
    //Get the two character prefix
    char* prefix = new char[2];
    for(int i=1; i>=0; i--) {
        prefix[i] = (prefix_int&byte_filter);
        prefix_int = prefix_int>>8;
    }
    return prefix;
}

int main(int argc, char* argv[])
{
    //Do a counter to see of how many characters are nonzero in the first
    //position.  We might only need 2 byte prefix.

    int non_zero = 0;
    uint64_t hash_start = 0;
    uint64_t hash_end = 16384;

    for(uint64_t hash_slot=hash_start; hash_slot<=hash_end; hash_slot++) {
        // Get a input to CRC16 that yields the hash slot
        uint64_t prefix_int = crc16_inverse(hash_slot);
        char* prefix = get_prefix(prefix_int);
        //std::cout<<"****"<<std::endl;
        //std::cout<<"hash_slot = "<<hash_slot<<std::endl;
        for(int i=0; i<2; i++) {
            //std::cout<<"prefix["<<i<<"] = "<<prefix[i]<<std::endl;
            //std::cout << std::bitset<8>(prefix[i])<<std::endl;
            if(prefix[i]==0) {
                std::cout<<"hash_slot "<<hash_slot<<" has zero!"<<std::endl;
            }
        }
        if(prefix[0]!=0)
            non_zero++;

        uint16_t crc_answer = crc16(prefix, 2);
        //std::cout<<"CRC answer = "<<crc_answer<<std::endl;
        if(crc_answer!=hash_slot)
            std::runtime_error("The values do not match for "+std::to_string(hash_slot));
        delete[] prefix;
    }
    std::cout<<"The number of non_zero first chars is "<<non_zero<<std::endl;
    return 0;
}