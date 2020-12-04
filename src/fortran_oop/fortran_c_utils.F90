module fortran_c_utils

use iso_c_binding, only : c_char, c_null_char

implicit none; private

public :: make_c_string

contains

function make_c_string( f_string ) result(c_string)
  character(len=*) :: f_string                             !< Fortran string to be converted
  character(kind=c_char) :: c_string(len_trim(f_string)+1) !< Output the C-string

  c_string = transfer(trim(f_string)//c_null_char, c_string)
end function make_c_string

end module fortran_c_utils