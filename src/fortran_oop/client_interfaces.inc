interface
  type(c_ptr) function c_constructor( cluster ) bind(c, name="SmartSimCClient")
    use iso_c_binding, only : c_ptr, c_bool
    logical(kind=c_bool) :: cluster !< True if a database cluster is being used
  end function c_constructor
end interface

interface
  subroutine c_destructor( client ) bind(c, name="DeleteCClient")
    use iso_c_binding, only : c_ptr
    type(c_ptr) :: client
  end subroutine
end interface
