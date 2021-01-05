subroutine initialize( this, name )
  class(dataset_type) :: this
  character(len=*)    :: name !< Name of the dataset

  integer(kind=c_size_t) :: name_length
  character(kind=c_char, len=len_trim(name)) :: c_name

  name_length = len_trim(name)
  c_name = name
  this%dataset = dataset_constructor(c_name, name_length)
end subroutine initialize
