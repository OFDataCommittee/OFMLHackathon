!> Get scalar metadata whose Fortran type is the equivalent 'int8' C-type
function get_meta_scalars_i8(this, name) result( meta )
  integer(kind=c_int8_t), pointer, dimension(:) :: meta
  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_i8

!> Get scalar metadata whose Fortran type is the equivalent 'int16' C-type
function get_meta_scalars_i16(this, name) result(meta)
  integer(kind=c_int16_t), pointer, dimension(:) :: meta
  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_i16

!> Get scalar metadata whose Fortran type is the equivalent 'int32' C-type
function get_meta_scalars_i32(this, name) result(meta)
  integer(kind=c_int32_t), pointer, dimension(:) :: meta
  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_i32

!> Get scalar metadata whose Fortran type is the equivalent 'int64' C-type
function get_meta_scalars_i64(this, name) result(meta)
  integer(kind=c_int64_t), pointer, dimension(:) :: meta
  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_i64

!> Get scalar metadata whose Fortran type is the equivalent 'float' C-type
function get_meta_scalars_float(this, name) result(meta)
  real(kind=c_float), pointer, dimension(:) :: meta
  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_float

!> Get scalar metadata whose Fortran type is the equivalent 'double' C-type
function get_meta_scalars_double(this, name) result(meta)
  real(kind=c_double), pointer, dimension(:) :: meta
  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_double