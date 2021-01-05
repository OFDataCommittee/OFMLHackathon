interface
  subroutine add_meta_scalar_c( dataset, name, name_length, data, data_type ) bind(c, name="add_meta_scalar")
    use iso_c_binding, only : c_ptr, c_size_t, c_char
    import :: enum_kind
    type(c_ptr),             value, intent(in) :: dataset     !< A c_ptr to the dataset object
    character(kind=c_char),         intent(in) :: name(*)     !< The name to use to place the tensor
    integer(kind=c_size_t),  value, intent(in) :: name_length !< The length of the name c-string,
                                                              !! excluding null terminating character
    type(c_ptr),             value, intent(in) :: data        !< A c ptr to the beginning of the data
    integer(kind=enum_kind), value, intent(in) :: data_type   !< The c-type of the data
  end subroutine add_meta_scalar_c
end interface

interface
  subroutine add_meta_string_c( dataset, name, name_length, data, data_length ) bind(c, name="add_meta_string")
    use iso_c_binding, only : c_ptr, c_size_t, c_char
    import :: enum_kind
    type(c_ptr),             value, intent(in) :: dataset     !< A c_ptr to the dataset object
    character(kind=c_char),         intent(in) :: name(*)     !< The name to use to place the tensor
    integer(kind=c_size_t),  value, intent(in) :: name_length !< The length of the name c-string,
                                                              !! excluding null terminating character
    character(kind=c_char),         intent(in) :: data(*)     !< The actual metadata string
    integer(kind=c_size_t),  value, intent(in) :: data_length !< Length of the metadata string
  end subroutine add_meta_string_c
end interface

interface
  type(c_ptr) function get_meta_scalars_c( dataset, name, name_length, length, data_type ) &
      bind(c, name="get_meta_scalars")
    use iso_c_binding, only : c_ptr, c_size_t, c_char
    import :: enum_kind
    type(c_ptr),            value, intent(in) :: dataset      !< A c_ptr to the dataset object
    character(kind=c_char),        intent(in) :: name(*)      !< The name to use to reference the metadata
    integer(kind=c_size_t), value, intent(in) :: name_length  !< The length of the name c-string,
                                                              !! excluding null terminating character
    integer(kind=c_size_t),        intent(  out) :: length    !< The number of entries in the field
    integer(kind=enum_kind),       intent(  out) :: data_type !< The c-type of the data
  end function get_meta_scalars_c
end interface