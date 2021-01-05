!> Store a dataset in the database
subroutine put_dataset( this, dataset )
  class(client_type), intent(in) :: this
  type(dataset_type), intent(in) :: dataset !< Dataset to store in the dataset

  call put_dataset_c(this%client_ptr, dataset%dataset)
end subroutine put_dataset

!> Retrieve a dataset from the database
type(dataset_type) function get_dataset( this, name )
  class(client_type) :: this
  character(len=*)   :: name !< Name of the dataset to get

  character(kind=c_char, len=len_trim(name)) :: c_name
  integer(kind=c_size_t) :: name_length

  c_name = trim(name)
  name_length = len_trim(name)
  get_dataset%dataset = get_dataset_c(this%client_ptr, c_name, name_length)
end function get_dataset

!> Rename a dataset stored in the database
subroutine rename_dataset( this, name, new_name )
  class(client_type) :: this
  character(len=*)   :: name     !< Original name of the dataset
  character(len=*)   :: new_name !< New name of the dataset

  character(kind=c_char, len=len_trim(name)) :: c_name
  character(kind=c_char, len=len_trim(new_name)) :: c_new_name
  integer(kind=c_size_t) :: name_length, new_name_length

  c_name = trim(name)
  c_new_name = trim(new_name)
  name_length = len_trim(name)
  new_name_length = len_trim(new_name)

  call rename_dataset_c(this%client_ptr, c_name, name_length, c_new_name, new_name_length)
end subroutine rename_dataset

!> Copy a dataset within the database to a new name
subroutine copy_dataset( this, name, new_name )
  class(client_type) :: this
  character(len=*)   :: name     !< Original name of the dataset
  character(len=*)   :: new_name !< New name of the dataset

  character(kind=c_char, len=len_trim(name)) :: c_name
  character(kind=c_char, len=len_trim(new_name)) :: c_new_name
  integer(kind=c_size_t) :: name_length, new_name_length

  c_name = trim(name)
  c_new_name = trim(new_name)
  name_length = len_trim(name)
  new_name_length = len_trim(new_name)

  call copy_dataset_c(this%client_ptr, c_name, name_length, c_new_name, new_name_length)
end subroutine copy_dataset

!> Delete a dataset stored within a database
subroutine delete_dataset( this, name )
  class(client_type) :: this
  character(len=*)   :: name !< Name of the dataset to delete

  character(kind=c_char, len=len_trim(name)) :: c_name
  integer(kind=c_size_t) :: name_length

  c_name = trim(name)
  name_length = len_trim(name)
  call delete_dataset_c(this%client_ptr, c_name, name_length)
end subroutine delete_dataset