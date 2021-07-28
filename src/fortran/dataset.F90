! BSD 2-Clause License
!
! Copyright (c) 2021, Hewlett Packard Enterprise
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

module smartredis_dataset

use iso_c_binding,   only : c_ptr, c_bool, c_null_ptr, c_char, c_int
use iso_c_binding,   only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t
use iso_c_binding,   only : c_loc, c_f_pointer

implicit none; private

include 'enums/enum_fortran.inc'
include 'dataset/dataset_interfaces.inc'
include 'dataset/add_tensor_interfaces.inc'
include 'dataset/unpack_dataset_tensor_interfaces.inc'
include 'dataset/metadata_interfaces.inc'

!> Contains multiple tensors and metadata used to describe an entire set of data
type, public :: dataset_type
  type(c_ptr) :: dataset_ptr !< A pointer to the initialized dataset object

  contains

  !> Initialize a new dataset with a given name
  procedure :: initialize => initialize_dataset
  !> Add metadata to the dataset with a given field and string
  procedure :: add_meta_string
  ! procedure :: get_meta_strings ! Not supported currently
  !> Add a tensor to be included as part of the dataset
  generic :: add_tensor => add_tensor_i8, add_tensor_i16, add_tensor_i32, add_tensor_i64, &
                           add_tensor_float, add_tensor_double
  !> Unpack a tensor that has previously been added to the dataset
  generic :: unpack_dataset_tensor => unpack_dataset_tensor_i8, unpack_dataset_tensor_i16, &
                                      unpack_dataset_tensor_i32, unpack_dataset_tensor_i64, &
                                      unpack_dataset_tensor_float, unpack_dataset_tensor_double
  !> Add metadata of type 'scalar' into a given field
  generic :: add_meta_scalar => add_meta_scalar_double, add_meta_scalar_float, add_meta_scalar_i32, add_meta_scalar_i64
  !> Retrieve scalar-type metadata as a vector
  generic :: get_meta_scalars => get_meta_scalars_double, get_meta_scalars_float, get_meta_scalars_i32, &
                                 get_meta_scalars_i64

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

subroutine initialize_dataset( self, name )
  class(dataset_type), intent(inout) :: self !< Receives the dataset
  character(len=*),    intent(in)    :: name !< Name of the dataset

  ! local variables
  integer(kind=c_size_t) :: name_length
  character(kind=c_char, len=len_trim(name)) :: c_name

  name_length = len_trim(name)
  c_name = trim(name)
  
  self%dataset_ptr = dataset_constructor(c_name, name_length)
end subroutine initialize_dataset

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int8' C-type
subroutine add_tensor_i8(self, name, data, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int8
  call add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
       c_fortran_contiguous)
end subroutine add_tensor_i8

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int16' C-type
subroutine add_tensor_i16(self, name, data, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int16
  call add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
       c_fortran_contiguous)
end subroutine add_tensor_i16

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int32' C-type
subroutine add_tensor_i32(self, name, data, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int32
  call add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
       c_fortran_contiguous)
end subroutine add_tensor_i32

!> Add a tensor to a dataset whose Fortran type is the equivalent 'int64' C-type
subroutine add_tensor_i64(self, name, data, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int64
  call add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
       c_fortran_contiguous)
end subroutine add_tensor_i64

!> Add a tensor to a dataset whose Fortran type is the equivalent 'float' C-type
subroutine add_tensor_float(self, name, data, dims)
  real(kind=c_float), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_flt
  call add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
       c_fortran_contiguous)
end subroutine add_tensor_float

!> Add a tensor to a dataset whose Fortran type is the equivalent 'double' C-type
subroutine add_tensor_double(self, name, data, dims)
  real(kind=c_double), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'dataset/add_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call add_tensor_c(self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
       c_fortran_contiguous)
end subroutine add_tensor_double


!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int8' C-type
subroutine unpack_dataset_tensor_i8(self, name, result, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(out) :: result !< Array to be populated with data
  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int8
  call unpack_dataset_tensor_c( self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i8

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int16' C-type
subroutine unpack_dataset_tensor_i16(self, name, result, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(out) :: result !< Array to be populated with data
  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int16
  call unpack_dataset_tensor_c( self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i16

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int32' C-type
subroutine unpack_dataset_tensor_i32(self, name, result, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(out) :: result !< Array to be populated with data
  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int32
  call unpack_dataset_tensor_c( self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i32

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'int64' C-type
subroutine unpack_dataset_tensor_i64(self, name, result, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(out) :: result !< Array to be populated with data
  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int64
  call unpack_dataset_tensor_c( self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_i64

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'float' C-type
subroutine unpack_dataset_tensor_float(self, name, result, dims)
  real(kind=c_float), dimension(..), target, intent(out) :: result !< Array to be populated with data
  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_flt
  call unpack_dataset_tensor_c( self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_float

!> Unpack a tensor into already allocated memory whose Fortran type is the equivalent 'double' C-type
subroutine unpack_dataset_tensor_double(self, name, result, dims)
  real(kind=c_double), dimension(..), target, intent(out) :: result !< Array to be populated with data
  include 'dataset/unpack_dataset_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call unpack_dataset_tensor_c( self%dataset_ptr, c_name, name_length, data_ptr, c_dims_ptr, c_n_dims, data_type, &
                                mem_layout )
end subroutine unpack_dataset_tensor_double


!> Get scalar metadata whose Fortran type is the equivalent 'int32' C-type
subroutine get_meta_scalars_i32(self, name, meta)
  class(dataset_type), intent(in) :: self !< The dataset
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int32_t), dimension(:), pointer :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind) :: expected_data_type = meta_int32
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_i32

!> Get scalar metadata whose Fortran type is the equivalent 'int64' C-type
subroutine get_meta_scalars_i64(self, name, meta)
  class(dataset_type), intent(in) :: self !< The dataset
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int64_t), dimension(:), pointer :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind) :: expected_data_type = meta_int64
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_i64

!> Get scalar metadata whose Fortran type is the equivalent 'float' C-type
subroutine get_meta_scalars_float(self, name, meta)
  class(dataset_type), intent(in) :: self !< The dataset
  character(len=*),    intent(in) :: name !< The name of the metadata field
  real(kind=c_float), dimension(:), pointer :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind) :: expected_data_type = meta_flt
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_float

!> Get scalar metadata whose Fortran type is the equivalent 'double' C-type
subroutine get_meta_scalars_double(self, name, meta)
  class(dataset_type), intent(in) :: self !< The dataset
  character(len=*),    intent(in) :: name !< The name of the metadata field
  real(kind=c_double), dimension(:), pointer :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind) :: expected_data_type = meta_dbl
  include 'dataset/get_meta_scalars_common.inc'
end subroutine get_meta_scalars_double

!> Add scalar metadata whose Fortran type is the equivalent 'int32' C-type
subroutine add_meta_scalar_i32(self, name, meta)
  class(dataset_type), intent(in) :: self !< The dataset
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int32_t), target, intent(in) :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_int32
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_i32

!> Add scalar metadata whose Fortran type is the equivalent 'int64' C-type
subroutine add_meta_scalar_i64(self, name, meta)
  class(dataset_type), intent(in) :: self !< The dataset
  character(len=*),    intent(in) :: name !< The name of the metadata field
  integer(kind=c_int64_t), target, intent(in) :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_int64
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_i64

!> Add scalar metadata whose Fortran type is the equivalent 'float' C-type
subroutine add_meta_scalar_float(self, name, meta)
  class(dataset_type),        intent(in) :: self !< The dataset
  character(len=*),           intent(in) :: name !< The name of the metadata field
  real(kind=c_float), target, intent(in) :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_flt
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_float

!> Add scalar metadata whose Fortran type is the equivalent 'double' C-type
subroutine add_meta_scalar_double(self, name, meta)
  class(dataset_type),         intent(in) :: self !< The dataset
  character(len=*),            intent(in) :: name !< The name of the metadata field
  real(kind=c_double), target, intent(in) :: meta !< The actual metadata

  ! local variables
  integer(kind=enum_kind), parameter :: meta_type = meta_dbl
  include 'dataset/add_meta_scalar_common.inc'
end subroutine add_meta_scalar_double

!> Add string-like metadata to the dataset
subroutine add_meta_string( self, name, meta )
  class(dataset_type), intent(in) :: self !< The dataset
  character(len=*),    intent(in) :: name !< The name of the metadata field
  character(len=*),    intent(in) :: meta !< The actual metadata

  ! local variables
  character(kind=c_char, len=len_trim(meta)) :: c_meta
  character(kind=c_char, len=len_trim(name)) :: c_name

  integer(kind=c_size_t) :: meta_length, name_length

  c_name = trim(name)
  c_meta = trim(meta)

  meta_length = len_trim(c_meta)
  name_length = len_trim(c_name)

  call add_meta_string_c( self%dataset_ptr, c_name, name_length, c_meta, meta_length )
end subroutine add_meta_string

end module smartredis_dataset
