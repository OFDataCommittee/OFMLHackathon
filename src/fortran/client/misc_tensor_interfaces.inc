interface
  subroutine rename_tensor_c(c_client, key, key_length, new_key, new_key_length) bind(c, name="rename_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value, intent(in) :: c_client   !< Pointer to the initialized client
    character(kind=c_char),        intent(in) :: key(*)     !< The key to use to place the tensor
    integer(kind=c_size_t), value, intent(in) :: key_length !< The length of the key c-string,
                                                            !! excluding null terminating character
    character(kind=c_char),        intent(in) :: new_key(*) !< The new tensor key
    integer(kind=c_size_t), value, intent(in) :: new_key_length !< The length of the new key c-string,
                                                                !! excluding null terminating character
  end subroutine rename_tensor_c
end interface

interface
  subroutine delete_tensor_c(c_client, key, key_length) bind(c, name="rename_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value, intent(in) :: c_client   !< Pointer to the initialized client
    character(kind=c_char),        intent(in) :: key(*)     !< The key of the tensor to delete
    integer(kind=c_size_t), value, intent(in) :: key_length !< The length of the key c-string,
  end subroutine delete_tensor_c
end interface

interface
  subroutine copy_tensor_c(c_client, src_name, src_name_length, dest_name, dest_name_length) &
      bind(c, name="copy_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value, intent(in) :: c_client        !< Pointer to the initialized client
    character(kind=c_char),        intent(in) :: src_name(*)     !< The key to use to place the tensor
    integer(kind=c_size_t), value, intent(in) :: src_name_length !< The length of the key c-string,
                                                                 !! excluding null terminating character
    character(kind=c_char),        intent(in) :: dest_name(*)    !< The new tensor key
    integer(kind=c_size_t), value, intent(in) :: dest_name_length !< The length of the new key c-string,
                                                                  !! excluding null terminating character
  end subroutine copy_tensor_c
end interface
