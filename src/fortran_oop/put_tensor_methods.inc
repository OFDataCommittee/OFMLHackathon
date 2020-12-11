!> Put a tensor whose Fortran type is the equivalent 'int8' C-type
subroutine put_tensor_i8(this, key, data, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'put_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int8
  call put_tensor_c(this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i8

!> Put a tensor whose Fortran type is the equivalent 'int16' C-type
subroutine put_tensor_i16(this, key, data, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'put_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int16
  call put_tensor_c(this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i16

!> Put a tensor whose Fortran type is the equivalent 'int32' C-type
subroutine put_tensor_i32(this, key, data, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'put_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int32
  call put_tensor_c(this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i32

!> Put a tensor whose Fortran type is the equivalent 'int64' C-type
subroutine put_tensor_i64(this, key, data, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'put_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int64
  call put_tensor_c(this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i64

!> Put a tensor whose Fortran type is the equivalent 'float' C-type
subroutine put_tensor_float(this, key, data, dims)
  real(kind=c_float), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'put_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_flt
  call put_tensor_c(this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_float

!> Put a tensor whose Fortran type is the equivalent 'double' C-type
subroutine put_tensor_double(this, key, data, dims)
  real(kind=c_double), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'put_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call put_tensor_c(this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_double