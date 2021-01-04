  character(kind=c_char)  :: c_name(len_trim(name))
  integer(kind=enum_kind) :: c_data_type, expected_data_type
  integer(kind=c_size_t) :: name_length
  type(c_ptr) :: meta_ptr

  c_name = transfer(trim(name),c_name)
  name_length = len_trim(name)
  meta_ptr = c_loc(meta)

  call add_meta_scalar_c(this%dataset, c_name, name_length, meta_ptr, meta_type)