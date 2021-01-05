  character(kind=c_char, len=len_trim(name))  :: c_name
  integer(kind=enum_kind) :: c_data_type
  integer(kind=c_size_t) :: name_length, meta_length
  type(c_ptr) :: result_ptr

  c_name = trim(name)
  name_length = len_trim(name)
  result_ptr = get_meta_scalars_c(this%dataset, c_name, name_length, meta_length, c_data_type)
  call c_f_pointer(result_ptr, meta, [meta_length])