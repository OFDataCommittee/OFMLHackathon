  class(client_type),                   intent(in) :: this  !< Pointer to the initialized client
  character(len=*),                     intent(in) :: key   !< The key to use to place the tensor
  integer, dimension(:),                intent(in) :: dims  !< Length along each dimension of the tensor

  ! Local variables
  integer(kind=c_size_t) :: ndim ! Number of dimensions
  type(c_ptr) :: data_ptr, c_dims_ptr
  character(kind=c_char) :: c_key(len_trim(key)) !< Transformed fortran 'key' to a c-string
  integer(kind=c_size_t) :: key_length, c_n_dims
  integer(kind=c_size_t), target :: c_dims(size(dims))
  integer(kind=enum_kind) :: data_type, mem_layout

  c_key = trim(key)
  key_length = len_trim(key)
  data_ptr = c_loc(result)
  c_dims(:) = dims(:)
  c_n_dims = size(dims)
  c_dims_ptr = c_loc(c_dims)
  mem_layout = c_fortran_contiguous