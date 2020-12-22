!> Move a tensor to a new key
subroutine rename_tensor(this, key, new_key)
  class(client_type), intent(in) :: this    !< The initialized Fortran SILC client
  character(len=*), intent(in) :: key     !< The key to use to place the tensor
                                          !! excluding null terminating character
  character(len=*), intent(in) :: new_key !< The new tensor key

  character(kind=c_char) :: c_key(len_trim(key))
  character(kind=c_char) :: c_new_key(len_trim(new_key))
  integer(kind=c_size_t) :: key_length, new_key_length

  c_key = transfer(trim(key),c_key)
  c_new_key = transfer(trim(new_key), c_new_key)

  key_length = len_trim(key)
  new_key_length = len_trim(new_key)

  call rename_tensor_c(this%client_ptr, c_key, key_length, c_new_key, new_key_length)

end subroutine rename_tensor

!> Delete a tensor
subroutine delete_tensor(this, key)
  class(client_type), intent(in) :: this !<  The initialized Fortran SILC client
  character(len=*), intent(in) :: key  !< The key to use to place the tensor

  character(kind=c_char) :: c_key(len_trim(key))
  integer(kind=c_size_t) :: key_length

  c_key = transfer(trim(key),c_key)
  key_length = len_trim(key)

  call delete_tensor_c(this%client_ptr, c_key, key_length)
end subroutine delete_tensor

!> Copy a tensor to the destination key
subroutine copy_tensor(this, src_name, dest_name)
  class(client_type), intent(in) :: this      !< The initialized Fortran SILC client
  character(len=*), intent(in) :: src_name  !< The key to use to place the tensor
                                            !! excluding null terminating character
  character(len=*), intent(in) :: dest_name !< The new tensor key

  character(kind=c_char) :: c_src_name(len_trim(src_name))
  character(kind=c_char) :: c_dest_name(len_trim(dest_name))
  integer(kind=c_size_t) :: src_name_length, dest_name_length

  c_src_name = transfer(trim(src_name), c_src_name)
  c_dest_name = transfer(trim(dest_name), c_dest_name)

  src_name_length = len_trim(src_name, kind=c_size_t)
  dest_name_length = len_trim(dest_name, kind=c_size_t)

  call copy_tensor_c(this%client_ptr, c_src_name, src_name_length, c_dest_name, dest_name_length)

end subroutine copy_tensor