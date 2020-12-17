module silc_client

use iso_c_binding, only : c_ptr, c_bool, c_null_ptr, c_char, c_int
use iso_c_binding, only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t
use iso_c_binding, only : c_loc, c_f_pointer

use silc_dataset, only : dataset_type

implicit none; private

include 'enum_fortran.inc'
include 'client/client_interfaces.inc'
include 'client/put_tensor_interfaces.inc'
include 'client/get_tensor_interfaces.inc'
include 'client/unpack_tensor_interfaces.inc'
include 'client/misc_tensor_interfaces.inc'
include 'client/model_interfaces.inc'
include 'client/script_interfaces.inc'

type, public :: client_type
  private

  logical(kind=c_bool) :: cluster = .false.        !< True if a database cluster is being used
  type(c_ptr)          :: client_ptr  = c_null_ptr !< Pointer to the initialized SmartSimClient

  contains

  ! Public procedures
  generic :: put_tensor => put_tensor_i8, put_tensor_i16, put_tensor_i32, put_tensor_i64, &
                           put_tensor_float, put_tensor_double
  generic :: unpack_tensor => unpack_tensor_i8, unpack_tensor_i16, unpack_tensor_i32, unpack_tensor_i64, &
                              unpack_tensor_float, unpack_tensor_double

  procedure :: initialize
  procedure :: destructor
  procedure :: key_exists
  procedure :: poll_key
  procedure :: get_tensor
  procedure :: rename_tensor
  procedure :: delete_tensor
  procedure :: copy_tensor
  procedure :: set_model_from_file
  procedure :: set_model
  procedure :: get_model
  procedure :: set_script_from_file
  procedure :: set_script
  procedure :: get_script
  procedure :: run_script
  procedure :: run_model

  ! Private procedures
  procedure, private :: put_tensor_i8
  procedure, private :: put_tensor_i16
  procedure, private :: put_tensor_i32
  procedure, private :: put_tensor_i64
  procedure, private :: put_tensor_float
  procedure, private :: put_tensor_double
  procedure, private :: unpack_tensor_i8
  procedure, private :: unpack_tensor_i16
  procedure, private :: unpack_tensor_i32
  procedure, private :: unpack_tensor_i64
  procedure, private :: unpack_tensor_float
  procedure, private :: unpack_tensor_double

  ! Not yet implemented
  ! procedure :: put_dataset           => client_put_dataset
  ! procedure :: get_dataset           => client_get_dataset
  ! procedure :: rename_dataset        => client_rename_dataset
  ! procedure :: copy_dataset          => client_copy_dataset
  ! procedure :: delete_dataset        => client_delete_dataset
end type client_type

contains

include 'client/client_methods.inc'
include 'client/put_tensor_methods.inc'
include 'client/get_tensor_methods.inc'
include 'client/unpack_tensor_methods.inc'
include 'client/misc_tensor_methods.inc'
include 'client/model_methods.inc'
include 'client/script_methods.inc'

end module silc_client