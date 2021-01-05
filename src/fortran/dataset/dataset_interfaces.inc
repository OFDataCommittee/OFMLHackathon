interface
  type(c_ptr) function dataset_constructor( name, name_length ) bind(c, name="CDataSet")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    character(kind=c_char) :: name !< Name of the dataset
    integer(kind=c_size_t), value :: name_length !< How many characters in the dataset's name
  end function dataset_constructor
end interface