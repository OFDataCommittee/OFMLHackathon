program test
  use iso_c_binding

  implicit none

  type, bind(c) :: foo
   integer(kind=c_int) :: id
   character(kind=c_char)   :: key(10)
   real(kind=c_float)  :: pi
   real(kind=c_double) :: one
  end type foo

  interface
    subroutine print_fortran_value(to_c) bind(c, name="print_foo_value")
      import
      type(foo), value :: to_c

    end subroutine print_fortran_value
  end interface

  interface
    subroutine print_fortran_pointer(to_c) bind(c, name="print_foo_ptr")
      import
      type(foo) :: to_c

    end subroutine print_fortran_pointer
  end interface


  type(foo) :: out

  out%id = 12
  out%key = transfer(trim("foo     !")//c_null_char, out%key)
  out%pi = 3.14
  out%one = 1.

  call print_fortran_value(out)
  call print_fortran_pointer(out)
end program test