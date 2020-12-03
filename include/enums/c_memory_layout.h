/* Defines the memory layout of
tensor data for c-clients to use
as a specifier.
*/

#ifndef SMARTSIM_CMEMORYLAYOUT_H
#define SMARTSIM_CMEMORYLAYOUT_H

typedef enum{
    c_nested=1,
    c_contiguous=2,
    c_fortran_nested=3,
    c_fortran_contiguous=4
}CMemoryLayout;

#endif //SMARTSIM_CMEMORYLAYOUT_H