!> Put a tensor whose Fortran type is the equivalent 'int8' C-type
subroutine unpack_tensor_i8(this, key, result, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'unpack_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int8
  call unpack_tensor_c( this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i8

!> Put a tensor whose Fortran type is the equivalent 'int16' C-type
subroutine unpack_tensor_i16(this, key, result, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'unpack_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int16
  call unpack_tensor_c( this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i16

!> Put a tensor whose Fortran type is the equivalent 'int32' C-type
subroutine unpack_tensor_i32(this, key, result, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'unpack_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int32
  call unpack_tensor_c( this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i32

!> Put a tensor whose Fortran type is the equivalent 'int64' C-type
subroutine unpack_tensor_i64(this, key, result, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'unpack_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int64
  call unpack_tensor_c( this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i64

!> Put a tensor whose Fortran type is the equivalent 'float' C-type
subroutine unpack_tensor_float(this, key, result, dims)
  real(kind=c_float), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'unpack_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_flt
  call unpack_tensor_c( this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_float

!> Put a tensor whose Fortran type is the equivalent 'double' C-type
subroutine unpack_tensor_double(this, key, result, dims)
  real(kind=c_double), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'unpack_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call unpack_tensor_c( this%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_double