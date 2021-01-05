  !** Beginning of code common to all put_tensor subroutines

  class(dataset_type),                   intent(in) :: this !< Fortran SLIC dataset
  character(len=*),                      intent(in) :: name  !< The unique name used to store in the database
  integer, dimension(:),                 intent(in) :: dims !< The length of each dimension
  ! Local variables
  integer(kind=c_size_t)                      :: c_n_dims ! Number of dimensions
  type(c_ptr) :: data_ptr, c_dims_ptr
  character(kind=c_char, len=len_trim(name)) :: c_name !< Transformed fortran 'name' to a c-string
  integer(kind=c_size_t) :: name_length
  integer(kind=c_size_t), target :: c_dims(size(dims))
  integer(kind=enum_kind) :: data_type

  ! Determine the shape of the array and the length of each dimension
  c_n_dims = size(dims)

  ! Create the pointer to the data array
  data_ptr = c_loc(data)

  ! Process the name and calculate its length
  c_name = trim(name)
  name_length = len_trim(name)

  c_dims(:) = dims(:)
  c_dims_ptr = c_loc(c_dims)

  !** End of code common to all put_tensor subroutines
