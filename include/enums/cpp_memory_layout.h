#ifndef SMARTSIM_MEMORYLAYOUT_H
#define SMARTSIM_MEMORYLAYOUT_H

#include "enums/c_memory_layout.h"

/* The MemoryLayout enum specifies
the layout of the memory space that
is provided or requested by the user.
*/
enum class MemoryLayout{
    nested = 1,
    contiguous = 2,
    fortran_contiguous = 3
};

inline MemoryLayout convert_layout(CMemoryLayout layout) {
  /* This function converts the CMemoryLayout to
  MemoryLayout.  This function is needed because
  of namespace limitations not allowed for the
  direct use of MemoryLayout.
  */

  switch(layout) {
    case(c_nested) :
      return MemoryLayout::nested;
      break;
    case(c_contiguous) :
      return MemoryLayout::contiguous;
      break;
    case(c_fortran_contiguous) :
      return MemoryLayout::fortran_contiguous;
      break;
    default :
      throw std::runtime_error("Unsupported enum "\
                               "conversion.");
  }
}

#endif //SMARTSIM_MEMORYLAYOUT_H