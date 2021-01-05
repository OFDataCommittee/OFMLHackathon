interface
  subroutine put_dataset_c( client, dataset ) bind(C, name="put_dataset")
    use iso_c_binding, only : c_ptr
    type(c_ptr), value, intent(in) :: client  !< Pointer to the initialized C-client
    type(c_ptr), value, intent(in) :: dataset !< Pointer to the dataset
  end subroutine put_dataset_c
end interface
interface
  type(c_ptr) function get_dataset_c( client, c_name, name_length ) bind(C, name="get_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value :: client !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Name of the dataset to retrieve from the database
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
  end function get_dataset_c
end interface
interface
  subroutine rename_dataset_c( client, c_name, name_length, c_new_name, new_name_length ) &
    bind(C, name="rename_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value :: client !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Original name of the dataset
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
    character(kind=c_char)        :: c_new_name(*) !< New name of the dataset to in the database
    integer(kind=c_size_t), value :: new_name_length !< Number of characters in the dataset's new name
  end subroutine rename_dataset_c
end interface
interface
  subroutine copy_dataset_c( client, c_name, name_length, c_new_name, new_name_length ) &
    bind(C, name="copy_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value :: client !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Original name of the dataset
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
    character(kind=c_char)        :: c_new_name(*) !< New name of the dataset to in the database
    integer(kind=c_size_t), value :: new_name_length !< Number of characters in the dataset's new name
  end subroutine copy_dataset_c
end interface
interface
  subroutine delete_dataset_c( client, c_name, name_length ) bind(C, name="delete_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value :: client    !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Name of the dataset to be deleted
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
  end subroutine delete_dataset_c
end interface



