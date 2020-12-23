module silc_dataset

use iso_c_binding,   only : c_ptr, c_bool, c_null_ptr, c_char, c_int
use iso_c_binding,   only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t
use iso_c_binding,   only : c_loc, c_f_pointer

implicit none; private

include 'enum_fortran.inc'
include 'dataset/dataset_interfaces.inc'
include 'dataset/add_tensor_interfaces.inc'
include 'dataset/get_dataset_tensor_interfaces.inc'
include 'dataset/unpack_dataset_tensor_interfaces.inc'
include 'dataset/metadata_interfaces.inc'

type, public :: dataset_type
  type(c_ptr) :: dataset !< A pointer to the initialized dataset object

  contains

  procedure :: initialize
  procedure :: get_dataset_tensor
  ! procedure :: add_meta_scalar
  ! procedure :: add_meta_string
  ! procedure :: get_meta_scalars
  ! procedure :: get_meta_strings ! Not supported currently

  generic :: add_tensor => add_tensor_i8, add_tensor_i16, add_tensor_i32, add_tensor_i64, &
                           add_tensor_float, add_tensor_double
  generic :: unpack_dataset_tensor => unpack_dataset_tensor_i8, unpack_dataset_tensor_i16, &
                                      unpack_dataset_tensor_i32, unpack_dataset_tensor_i64, &
                                      unpack_dataset_tensor_float, unpack_dataset_tensor_double

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

end type dataset_type

contains

include 'dataset/dataset_methods.inc'
include 'dataset/add_tensor_methods.inc'
include 'dataset/get_dataset_tensor_methods.inc'
include 'dataset/unpack_dataset_tensor_methods.inc'
include 'dataset/metadata_methods.inc'

end module silc_dataset