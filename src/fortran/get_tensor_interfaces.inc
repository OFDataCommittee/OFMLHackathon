interface
  subroutine get_tensor_c(c_client, key, key_length, data, dims, n_dims, data_type, mem_layout) &
      bind(c, name="get_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind

    type(c_ptr),             value, intent(in   ) :: c_client   !< Pointer to the initialized client
    character(kind=c_char),         intent(in   ) :: key(*)     !< The key to use to place the tensor
    integer(kind=c_size_t),  value, intent(in   ) :: key_length !< The length of the key c-gtgtstring,
                                                                !! excluding null terminating character
    type(c_ptr),                    intent(  out) :: data       !< A c ptr to the beginning of the data
    type(c_ptr),                    intent(  out) :: dims       !< Length of each dimension
    integer(kind=c_size_t),         intent(  out) :: n_dims     !< The number of dimensions
    type(c_ptr),                    intent(  out) :: data_type  !< The data type of the tensor
    integer(kind=enum_kind), value, intent(in   ) :: mem_layout !< The memory layout of the data
  end subroutine get_tensor_c
end interface