/* Defines the memory layout of
tensor data for c-clients to use
as a specifier.
*/

#ifndef SMARTREDIS_CMEMORYLAYOUT_H
#define SMARTREDIS_CMEMORYLAYOUT_H

typedef enum{
    c_nested=1,
    c_contiguous=2,
    c_fortran_nested=3,
    c_fortran_contiguous=4
}CMemoryLayout;

#endif //SMARTREDIS_CMEMORYLAYOUT_H