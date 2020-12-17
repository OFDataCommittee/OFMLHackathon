!> Add a tensor to a dataset whose Fortran type is the equivalent 'int8' C-type
subroutine add_tensor_i8(this, name, data, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int8
  call add_tensor_c(this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine add_tensor_i8

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int16' C-type
subroutine add_tensor_i16(this, name, data, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int16
  call add_tensor_c(this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine add_tensor_i16

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int32' C-type
subroutine add_tensor_i32(this, name, data, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int32
  call add_tensor_c(this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine add_tensor_i32

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int64' C-type
subroutine add_tensor_i64(this, name, data, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int64
  call add_tensor_c(this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine add_tensor_i64

!> Add a tensor to a dataset whose Fortran type is the equivalent 'float' C-type
subroutine add_tensor_float(this, name, data, dims)
  real(kind=c_float), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_flt
  call add_tensor_c(this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine add_tensor_float

!> Add a tensor to a dataset whose Fortran type is the equivalent 'double' C-type
subroutine add_tensor_double(this, name, data, dims)
  real(kind=c_double), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call add_tensor_c(this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine add_tensor_double