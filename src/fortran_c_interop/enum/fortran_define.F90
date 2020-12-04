program main
  use iso_c_binding

  implicit none

  enum, bind(c)
    enumerator :: color = 0
    enumerator :: red  = 255
    enumerator :: blue = 1
  end enum

  interface
    subroutine print_enum( thecolor ) bind(c, name="print_enum_c")
      integer(kind(color)), value :: thecolor
    end subroutine
  end interface

  interface
    integer(kind(color)) function set_enum( ) bind(c, name="set_enum_c")
    end function
  end interface

  integer(kind(color)) :: my_color, their_color

  my_color = red

  call print_enum( my_color )
  their_color = set_enum()
  print *, their_color
end program main