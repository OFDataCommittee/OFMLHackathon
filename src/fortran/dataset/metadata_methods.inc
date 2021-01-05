!> Get scalar metadata whose Fortran type is the equivalent 'int32' C-type
subroutine get_meta_scalars_i32(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int32_t), dimension(:), pointer :: meta !< The actual metadata

  integer(kind=enum_kind) :: expected_data_type = meta_int32
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_i32

!> Get scalar metadata whose Fortran type is the equivalent 'int64' C-type
subroutine get_meta_scalars_i64(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int64_t), dimension(:), pointer :: meta !< The actual metadata

  integer(kind=enum_kind) :: expected_data_type = meta_int64
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_i64

!> Get scalar metadata whose Fortran type is the equivalent 'float' C-type
subroutine get_meta_scalars_float(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  real(kind=c_float), dimension(:), pointer :: meta !< The actual metadata

  integer(kind=enum_kind) :: expected_data_type = meta_flt
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_float

!> Get scalar metadata whose Fortran type is the equivalent 'double' C-type
subroutine get_meta_scalars_double(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  real(kind=c_double), dimension(:), pointer :: meta !< The actual metadata

  integer(kind=enum_kind) :: expected_data_type = meta_dbl
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_double

!> Add scalar metadata whose Fortran type is the equivalent 'int32' C-type
subroutine add_meta_scalar_i32(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int32_t), target, intent(in) :: meta !< The actual metadata

  integer(kind=enum_kind), parameter :: meta_type = meta_int32
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_i32

!> Add scalar metadata whose Fortran type is the equivalent 'int64' C-type
subroutine add_meta_scalar_i64(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int64_t), target, intent(in) :: meta !< The actual metadata

  integer(kind=enum_kind), parameter :: meta_type = meta_int64
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_i64

!> Add scalar metadata whose Fortran type is the equivalent 'float' C-type
subroutine add_meta_scalar_float(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  real(kind=c_float), target, intent(in) :: meta !< The actual metadata

  integer(kind=enum_kind), parameter :: meta_type = meta_flt
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_float

!> Add scalar metadata whose Fortran type is the equivalent 'double' C-type
subroutine add_meta_scalar_double(this, name, meta)
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  real(kind=c_double), target, intent(in) :: meta !< The actual metadata

  integer(kind=enum_kind), parameter :: meta_type = meta_dbl
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_double

!> Add string-like metadata to the dataset
subroutine add_meta_string( this, name, meta )
  class(dataset_type), intent(in) :: this
  character(len=*),    intent(in) :: name !< The name of the metadata field
  character(len=*),    intent(in) :: meta !< The actual metadata

  character(kind=c_char, len=len_trim(meta)) :: c_meta
  character(kind=c_char, len=len_trim(name)) :: c_name

  integer(kind=c_size_t) :: meta_length, name_length

  c_name = name
  c_meta = meta

  meta_length = len(c_meta)
  name_length = len(c_name)

  call add_meta_string_c( this%dataset, c_name, name_length, c_meta, meta_length )

end subroutine add_meta_string