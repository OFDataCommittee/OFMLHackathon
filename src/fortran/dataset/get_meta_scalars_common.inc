  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name

  character(kind=c_char)  :: c_name(len_trim(name))
  integer(kind=enum_kind) :: c_data_type
  integer(kind=c_size_t) :: name_length, meta_length
  type(c_ptr) :: result_ptr

  c_name = trim(name)
  name_length = len(c_name)
  result_ptr = get_meta_scalars_c(this%dataset, c_name, name_length, meta_length, c_data_type)
  call c_f_pointer(result_ptr, meta, [meta_length])