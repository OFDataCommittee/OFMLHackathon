!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int8' C-type
subroutine unpack_dataset_tensor_i8(this, name, result, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'dataset/unpack_dataset_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int8
  call unpack_dataset_tensor_c( this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i8

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int16' C-type
subroutine unpack_dataset_tensor_i16(this, name, result, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'dataset/unpack_dataset_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int16
  call unpack_dataset_tensor_c( this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i16

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int32' C-type
subroutine unpack_dataset_tensor_i32(this, name, result, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'dataset/unpack_dataset_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int32
  call unpack_dataset_tensor_c( this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i32

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int64' C-type
subroutine unpack_dataset_tensor_i64(this, name, result, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'dataset/unpack_dataset_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_int64
  call unpack_dataset_tensor_c( this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i64

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'float' C-type
subroutine unpack_dataset_tensor_float(this, name, result, dims)
  real(kind=c_float), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'dataset/unpack_dataset_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_flt
  call unpack_dataset_tensor_c( this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_float

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'double' C-type
subroutine unpack_dataset_tensor_double(this, name, result, dims)
  real(kind=c_double), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'dataset/unpack_dataset_tensor_methods_common.inc'
  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call unpack_dataset_tensor_c( this%dataset, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_double