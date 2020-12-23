interface
  type(c_ptr) function dataset_constructor( ) bind(c, name="CDataSet")
    use iso_c_binding, only : c_ptr
  end function dataset_constructor
end interface