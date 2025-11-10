! BSD 2-Clause License
!
! Copyright (c) 2021-2023, Hewlett Packard Enterprise
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

! Note the below macros are here to allow compilation with Nvidia drivers
! While assumed size should be sufficient, this does not seem to work with
! Intel and GNU (however those have support for assumed rank)
#ifdef __NVCOMPILER
#define DIM_RANK_SPEC dimension(*)
#else
#define DIM_RANK_SPEC dimension(..)
#endif

module smartredis_dataset

use iso_c_binding,   only : c_ptr, c_char, c_int
use iso_c_binding,   only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t
use iso_c_binding,   only : c_loc, c_f_pointer

use, intrinsic :: iso_fortran_env, only: stderr => error_unit

use fortran_c_interop, only : enum_kind

implicit none; private

include 'enum_fortran.inc'
include 'dataset/dataset_interfaces.inc'
include 'dataset/tensor_interfaces.inc'
include 'dataset/unpack_dataset_tensor_interfaces.inc'
include 'dataset/metadata_interfaces.inc'
#include "errors/errors_interfaces.inc"

public :: enum_kind !< The kind of integer equivalent to a C enum. According to C an Fortran
                    !! standards this should be c_int, but is renamed here to ensure that
                    !! users do not have to import the iso_c_binding module into their
                    !! programs

!> Contains multiple tensors and metadata used to describe an entire set of data
type, public :: dataset_type
  type(c_ptr) :: dataset_ptr !< A pointer to the initialized dataset object

  contains

  !> Initialize a new dataset with a given name
  procedure :: initialize => initialize_dataset
  !> Access the raw C pointer for the dataset
  procedure :: get_c_pointer

  ! Metadata procedures
  !> Add metadata to the dataset with a given field and string
  procedure :: add_meta_string
  ! Retrieve the strings associated with a metadata string field
  !> procedure :: get_meta_strings ! Not supported currently
  !> Add metadata of type 'scalar' into a given field
  generic :: add_meta_scalar => add_meta_scalar_double, add_meta_scalar_float, add_meta_scalar_i32, add_meta_scalar_i64
  !> Retrieve scalar-type metadata as a vector
  generic :: get_meta_scalars => get_meta_scalars_double, get_meta_scalars_float, get_meta_scalars_i32, &
                                 get_meta_scalars_i64
  ! Retrieve the names of metadata fields
  !> procedure :: get_metadata_field_names ! Not supported currently
  !> Retrieve the type for a metadata field
  procedure :: get_metadata_field_type

  ! Tensor procedures
  !> Add a tensor to be included as part of the dataset
  generic :: add_tensor => add_tensor_i8, add_tensor_i16, add_tensor_i32, add_tensor_i64, &
                           add_tensor_float, add_tensor_double
  !> Unpack a tensor that has previously been added to the dataset
  generic :: unpack_dataset_tensor => unpack_dataset_tensor_i8, unpack_dataset_tensor_i16, &
                                      unpack_dataset_tensor_i32, unpack_dataset_tensor_i64, &
                                      unpack_dataset_tensor_float, unpack_dataset_tensor_double
  ! Retrieve the names of tensors
  !> procedure :: get_tensor_names ! Not supported currently
  !> Retrieve the type for a tensor
  procedure :: get_tensor_type
  !> Retrieve the dimensions for a tensor
  procedure :: get_tensor_dims
  !> Retrieve a string representation of the dataset
  procedure :: to_string
  !> Print a string representation of the dataset
  procedure :: print_dataset

  ! Private procedures
  procedure, private :: add_tensor_i8
  procedure, private :: add_tensor_i16
  procedure, private :: add_tensor_i32
  procedure, private :: add_tensor_i64
  procedure, private :: add_tensor_float
  procedure, private :: add_tensor_double
  procedure, private :: unpack_dataset_tensor_i8
  procedure, private :: unpack_dataset_tensor_i16
  procedure, private :: unpack_dataset_tensor_i32
  procedure, private :: unpack_dataset_tensor_i64
  procedure, private :: unpack_dataset_tensor_float
  procedure, private :: unpack_dataset_tensor_double
  procedure, private :: add_meta_scalar_double
  procedure, private :: add_meta_scalar_float
  procedure, private :: add_meta_scalar_i32
  procedure, private :: add_meta_scalar_i64
  procedure, private :: get_meta_scalars_double
  procedure, private :: get_meta_scalars_float
  procedure, private :: get_meta_scalars_i32
  procedure, private :: get_meta_scalars_i64
end type dataset_type

contains


!> Initialize the dataset
function initialize_dataset(self, name) result(code)
  class(dataset_type), intent(inout) :: self !< Receives the dataset
  character(len=*),    intent(in)    :: name !< Name of the dataset
  integer(kind=enum_kind)            :: code !< Result of the operation

  ! Local variables
  integer(kind=c_size_t) :: name_length
  character(kind=c_char, len=len_trim(name)) :: c_name

  name_length = len_trim(name)
  c_name = trim(name)

  code = dataset_constructor(c_name, name_length, self%dataset_ptr)
end function initialize_dataset

!> Access the raw C pointer for the dataset
function get_c_pointer(self)
  type(c_ptr)                     :: get_c_pointer
  class(dataset_type), intent(in) :: self
  get_c_pointer = self%dataset_ptr
end function get_c_pointer

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int8' C-type
function add_tensor_i8(self, name, data, dims) result(code)
  integer(kind=c_int8_t), DIM_RANK_SPEC, target, intent(in) :: data !< Data to be sent
  class(dataset_type),   intent(in)  :: self !< Fortran SmartRedis dataset
  character(len=*),      intent(in)  :: name !< The unique name used to store in the database
  integer, dimension(:), intent(in)  :: dims !< The length of each dimension
  integer(kind=enum_kind)            :: code !< Result of the operation

  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int8
  code = add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, &
       c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end function add_tensor_i8

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int16' C-type
function add_tensor_i16(self, name, data, dims) result(code)
  integer(kind=c_int16_t), DIM_RANK_SPEC, target, intent(in) :: data !< Data to be sent
  class(dataset_type),   intent(in)  :: self !< Fortran SmartRedis dataset
  character(len=*),      intent(in)  :: name !< The unique name used to store in the database
  integer, dimension(:), intent(in)  :: dims !< The length of each dimension
  integer(kind=enum_kind)            :: code !< Result of the operation

  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int16
  code = add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, &
       c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end function add_tensor_i16

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int32' C-type
function add_tensor_i32(self, name, data, dims) result(code)
  integer(kind=c_int32_t), DIM_RANK_SPEC, target, intent(in) :: data !< Data to be sent
  class(dataset_type),   intent(in)  :: self !< Fortran SmartRedis dataset
  character(len=*),      intent(in)  :: name !< The unique name used to store in the database
  integer, dimension(:), intent(in)  :: dims !< The length of each dimension
  integer(kind=enum_kind)            :: code !< Result of the operation

  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int32
  code = add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, &
       c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end function add_tensor_i32

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int64' C-type
function add_tensor_i64(self, name, data, dims) result(code)
  integer(kind=c_int64_t), DIM_RANK_SPEC, target, intent(in) :: data !< Data to be sent
  class(dataset_type),   intent(in)  :: self !< Fortran SmartRedis dataset
  character(len=*),      intent(in)  :: name !< The unique name used to store in the database
  integer, dimension(:), intent(in)  :: dims !< The length of each dimension
  integer(kind=enum_kind)            :: code !< Result of the operation

  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int64
  code = add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, &
       c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end function add_tensor_i64

!> Add a tensor to a dataset whose Fortran type is the equivalent 'float' C-type
function add_tensor_float(self, name, data, dims) result(code)
  real(kind=c_float), DIM_RANK_SPEC, target, intent(in) :: data !< Data to be sent
  class(dataset_type),   intent(in)  :: self !< Fortran SmartRedis dataset
  character(len=*),      intent(in)  :: name !< The unique name used to store in the database
  integer, dimension(:), intent(in)  :: dims !< The length of each dimension
  integer(kind=enum_kind)            :: code !< Result of the operation

  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_flt
  code = add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, &
       c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end function add_tensor_float

!> Add a tensor to a dataset whose Fortran type is the equivalent 'double' C-type
function add_tensor_double(self, name, data, dims) result(code)
  real(kind=c_double), DIM_RANK_SPEC, target, intent(in) :: data !< Data to be sent
  class(dataset_type),   intent(in)  :: self !< Fortran SmartRedis dataset
  character(len=*),      intent(in)  :: name !< The unique name used to store in the database
  integer, dimension(:), intent(in)  :: dims !< The length of each dimension
  integer(kind=enum_kind)            :: code !< Result of the operation

  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_dbl
  code = add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, &
       c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end function add_tensor_double


!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int8' C-type
function unpack_dataset_tensor_i8(self, name, result, dims) result(code)
  integer(kind=c_int8_t), DIM_RANK_SPEC, target, intent(out) :: result !< Array to be populated with data
  class(dataset_type),                  intent(in) :: self !< Pointer to the initialized dataset
  character(len=*),                     intent(in) :: name !< The name to use to place the tensor
  integer, dimension(:),                intent(in) :: dims !< Length along each dimension of the tensor
  integer(kind=enum_kind)                          :: code

  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int8
  code = unpack_dataset_tensor_c(self%dataset_ptr, c_name, name_length, &
       data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout)
end function unpack_dataset_tensor_i8

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int16' C-type
function unpack_dataset_tensor_i16(self, name, result, dims) result(code)
  integer(kind=c_int16_t), DIM_RANK_SPEC, target, intent(out) :: result !< Array to be populated with data
  class(dataset_type),                  intent(in) :: self !< Pointer to the initialized dataset
  character(len=*),                     intent(in) :: name !< The name to use to place the tensor
  integer, dimension(:),                intent(in) :: dims !< Length along each dimension of the tensor
  integer(kind=enum_kind)                          :: code

  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int16
  code = unpack_dataset_tensor_c(self%dataset_ptr, c_name, name_length, &
       data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout)
end function unpack_dataset_tensor_i16

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int32' C-type
function unpack_dataset_tensor_i32(self, name, result, dims) result(code)
  integer(kind=c_int32_t), DIM_RANK_SPEC, target, intent(out) :: result !< Array to be populated with data
  class(dataset_type),                  intent(in) :: self !< Pointer to the initialized dataset
  character(len=*),                     intent(in) :: name !< The name to use to place the tensor
  integer, dimension(:),                intent(in) :: dims !< Length along each dimension of the tensor
  integer(kind=enum_kind)                          :: code

  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int32
  code = unpack_dataset_tensor_c(self%dataset_ptr, c_name, name_length, &
       data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout)
end function unpack_dataset_tensor_i32

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int64' C-type
function unpack_dataset_tensor_i64(self, name, result, dims) result(code)
  integer(kind=c_int64_t), DIM_RANK_SPEC, target, intent(out) :: result !< Array to be populated with data
  class(dataset_type),                  intent(in) :: self !< Pointer to the initialized dataset
  character(len=*),                     intent(in) :: name !< The name to use to place the tensor
  integer, dimension(:),                intent(in) :: dims !< Length along each dimension of the tensor
  integer(kind=enum_kind)                          :: code

  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int64
  code = unpack_dataset_tensor_c(self%dataset_ptr, c_name, name_length, &
       data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout)
end function unpack_dataset_tensor_i64

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'float' C-type
function unpack_dataset_tensor_float(self, name, result, dims) result(code)
  real(kind=c_float), DIM_RANK_SPEC, target, intent(out) :: result !< Array to be populated with data
  class(dataset_type),                  intent(in) :: self !< Pointer to the initialized dataset
  character(len=*),                     intent(in) :: name !< The name to use to place the tensor
  integer, dimension(:),                intent(in) :: dims !< Length along each dimension of the tensor
  integer(kind=enum_kind)                          :: code

  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_flt
  code = unpack_dataset_tensor_c(self%dataset_ptr, c_name, name_length, &
       data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout)
end function unpack_dataset_tensor_float

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'double' C-type
function unpack_dataset_tensor_double(self, name, result, dims) result(code)
  real(kind=c_double), DIM_RANK_SPEC, target, intent(out) :: result !< Array to be populated with data
  class(dataset_type),                  intent(in) :: self !< Pointer to the initialized dataset
  character(len=*),                     intent(in) :: name !< The name to use to place the tensor
  integer, dimension(:),                intent(in) :: dims !< Length along each dimension of the tensor
  integer(kind=enum_kind)                          :: code

  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_dbl
  code = unpack_dataset_tensor_c(self%dataset_ptr, c_name, name_length, &
       data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout)
end function unpack_dataset_tensor_double


!> Get scalar metadata whose Fortran type is the equivalent 'int32' C-type
function get_meta_scalars_i32(self, name, meta) result(code)
  class(dataset_type),                intent(in) :: self !< The dataset
  character(len=*),                   intent(in) :: name !< The name of the metadata field
  integer(kind=c_int32_t), dimension(:), pointer :: meta !< The actual metadata
  integer(kind=enum_kind)                        :: code !< Result of the operation

  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_i32

!> Get scalar metadata whose Fortran type is the equivalent 'int64' C-type
function get_meta_scalars_i64(self, name, meta) result(code)
  class(dataset_type),                intent(in) :: self !< The dataset
  character(len=*),                   intent(in) :: name !< The name of the metadata field
  integer(kind=c_int64_t), dimension(:), pointer :: meta !< The actual metadata
  integer(kind=enum_kind)                        :: code !< Result of the operation

  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_i64

!> Get scalar metadata whose Fortran type is the equivalent 'float' C-type
function get_meta_scalars_float(self, name, meta) result(code)
  class(dataset_type),           intent(in) :: self !< The dataset
  character(len=*),              intent(in) :: name !< The name of the metadata field
  real(kind=c_float), dimension(:), pointer :: meta !< The actual metadata
  integer(kind=enum_kind)                   :: code !< Result of the operation

  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_float

!> Get scalar metadata whose Fortran type is the equivalent 'double' C-type
function get_meta_scalars_double(self, name, meta) result(code)
  class(dataset_type),            intent(in) :: self !< The dataset
  character(len=*),               intent(in) :: name !< The name of the metadata field
  real(kind=c_double), dimension(:), pointer :: meta !< The actual metadata
  integer(kind=enum_kind)                    :: code !< Result of the operation

  include 'dataset/get_meta_scalars_common.inc'
end function get_meta_scalars_double

!> Add scalar metadata whose Fortran type is the equivalent 'int32' C-type
function add_meta_scalar_i32(self, name, meta) result(code)
  class(dataset_type),             intent(in) :: self !< The dataset
  character(len=*),                intent(in) :: name !< The name of the metadata field
  integer(kind=c_int32_t), target, intent(in) :: meta !< The actual metadata
  integer(kind=enum_kind)                     :: code !< Result of the operation

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_int32
  include 'dataset/add_meta_scalar_common.inc'
end function add_meta_scalar_i32

!> Add scalar metadata whose Fortran type is the equivalent 'int64' C-type
function add_meta_scalar_i64(self, name, meta) result(code)
  class(dataset_type),             intent(in) :: self !< The dataset
  character(len=*),                intent(in) :: name !< The name of the metadata field
  integer(kind=c_int64_t), target, intent(in) :: meta !< The actual metadata
  integer(kind=enum_kind)                     :: code !< Result of the operation

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_int64
  include 'dataset/add_meta_scalar_common.inc'
end function add_meta_scalar_i64

!> Add scalar metadata whose Fortran type is the equivalent 'float' C-type
function add_meta_scalar_float(self, name, meta) result(code)
  class(dataset_type),        intent(in) :: self !< The dataset
  character(len=*),           intent(in) :: name !< The name of the metadata field
  real(kind=c_float), target, intent(in) :: meta !< The actual metadata
  integer(kind=enum_kind)                :: code !< Result of the operation

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_flt
  include 'dataset/add_meta_scalar_common.inc'
end function add_meta_scalar_float

!> Add scalar metadata whose Fortran type is the equivalent 'double' C-type
function add_meta_scalar_double(self, name, meta) result(code)
  class(dataset_type),         intent(in) :: self !< The dataset
  character(len=*),            intent(in) :: name !< The name of the metadata field
  real(kind=c_double), target, intent(in) :: meta !< The actual metadata
  integer(kind=enum_kind)                 :: code !< Result of the operation

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_dbl
  include 'dataset/add_meta_scalar_common.inc'
end function add_meta_scalar_double

!> Add string-like metadata to the dataset
function add_meta_string(self, name, meta) result(code)
  class(dataset_type),     intent(in) :: self !< The dataset
  character(len=*),        intent(in) :: name !< The name of the metadata field
  character(len=*),        intent(in) :: meta !< The actual metadata
  integer(kind=enum_kind)             :: code !< Result of the operation

  ! local variables
  character(kind=c_char, len=len_trim(meta)) :: c_meta
  character(kind=c_char, len=len_trim(name)) :: c_name

  integer(kind=c_size_t) :: meta_length, name_length

  c_name = trim(name)
  c_meta = trim(meta)

  meta_length = len_trim(c_meta)
  name_length = len_trim(c_name)

  code = add_meta_string_c(self%dataset_ptr, c_name, name_length, c_meta, meta_length)
end function add_meta_string

!> Retrieve the type for a metadata field
function get_metadata_field_type(self, name, mdtype) result(code)
  class(dataset_type),     intent(in)  :: self   !< The dataset
  character(len=*),        intent(in)  :: name   !< The name of the metadata field
  integer(kind=enum_kind), intent(out) :: mdtype !< Receives the type
  integer(kind=enum_kind)              :: code   !< Result of the operation

  ! local variables
  character(kind=c_char, len=len_trim(name)) :: c_name
  integer(kind=c_size_t) :: name_length

  c_name = trim(name)
  name_length = len_trim(c_name)

  code = get_metadata_field_type_c(self%dataset_ptr, c_name, name_length, mdtype)
end function get_metadata_field_type

!> Retrieve the type for a tensor
function get_tensor_type(self, name, ttype) result(code)
  class(dataset_type),     intent(in)  :: self  !< The dataset
  character(len=*),        intent(in)  :: name  !< The name of the tensor
  integer(kind=enum_kind), intent(out) :: ttype !< Receives the type
  integer(kind=enum_kind)              :: code  !< Result of the operation

  ! local variables
  character(kind=c_char, len=len_trim(name)) :: c_name
  integer(kind=c_size_t) :: name_length

  c_name = trim(name)
  name_length = len_trim(c_name)

  code = get_tensor_type_c(self%dataset_ptr, c_name, name_length, ttype)
end function get_tensor_type


!> Retrieve the dimensions for a tensor into a supplied buffer, or receive the
!! number of dimensions if the supplied buffer is too small
function get_tensor_dims(self, name, dims, dims_length) result(code)
  class(dataset_type),     intent(in)    :: self  !< The dataset
  character(len=*),        intent(in)    :: name  !< The name of the tensor
  integer, dimension(:), target, intent(inout) :: dims !< Receives the tensor dimensions
  integer,  intent(inout) :: dims_length !< Receives the number of tensor dimensions
  integer(kind=enum_kind)                :: code  !< Result of the operation

  ! local variables
  character(kind=c_char, len=len_trim(name)) :: c_name
  integer(kind=c_size_t) :: name_length
  type(c_ptr) :: dims_ptr
  integer(kind=c_size_t), dimension(size(dims)), target :: c_dims
  integer(kind=c_size_t) ::  c_dims_length

  c_name = trim(name)
  name_length = len_trim(c_name)

  if (dims_length .gt. size(dims)) then
    error stop 'dims_length .gt. size(dims) in call to get_tensor_dims'
  end if
  dims_ptr = c_loc(c_dims)
  c_dims_length = dims_length

  code = get_tensor_dims_c(self%dataset_ptr, c_name, name_length, dims_ptr, c_dims_length)
  dims = int(c_dims, kind(dims))
  dims_length = int(c_dims_length, kind(dims_length))
end function get_tensor_dims


!> Retrieve a string representation of the dataset
function to_string(self)
  character(kind=c_char, len=:), allocatable :: to_string !< Text version of dataset
  class(dataset_type),     intent(in)        :: self      !< The dataset

  type(c_ptr)                                :: c_ds_str
  integer(kind=c_size_t)                     :: c_ds_str_len

  ! Get the string representation of the dataset from C
  c_ds_str = dataset_to_string_c(self%dataset_ptr)
  c_ds_str_len = c_strlen(c_ds_str)
  to_string = make_str(c_ds_str, c_ds_str_len)
end function to_string

!> Convert a pointer view of a string to a Fortran string
function make_str(strptr, str_len)
  character(kind=c_char, len=:), allocatable :: make_str
  type(c_ptr), intent(in), value             :: strptr
  integer(kind=c_size_t)                     :: str_len

  character(len=str_len, kind=c_char), pointer :: ptrview
  call c_f_pointer(strptr, ptrview)
  make_str = ptrview
end function make_str

!> Print a string representation of the dataset
subroutine print_dataset(self, unit)
  class(dataset_type), intent(in)  :: self  !< The dataset
  integer, optional,   intent(in)  :: unit !< Unit to which to print the dataset

  ! Determine which unit to write to
  integer :: target_unit
  target_unit = STDERR
  if (present(unit)) target_unit = unit

  ! Write the error to the target unit
  write(target_unit,*) to_string(self)
end subroutine print_dataset

end module smartredis_dataset
