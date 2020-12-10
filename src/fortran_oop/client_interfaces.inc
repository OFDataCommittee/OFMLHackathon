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

interface
  function key_exists_c( client, key, key_length ) bind(c, name="key_exists")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t
    logical(kind=c_bool)          :: key_exists_c
    type(c_ptr),            value :: client
    character(kind=c_char)        :: key(*)
    integer(kind=c_size_t), value :: key_length
  end function key_exists_c
end interface

interface
  function poll_key_c( client, key, key_length, poll_frequency_ms, num_tries ) bind(c, name="poll_key")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    logical(kind=c_bool)          :: poll_key_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: key(*)
    integer(kind=c_size_t), value :: key_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries

  end function poll_key_c
end interface