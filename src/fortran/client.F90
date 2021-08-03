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

module smartredis_client

use iso_c_binding, only : c_ptr, c_bool, c_null_ptr, c_char, c_int
use iso_c_binding, only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t
use iso_c_binding, only : c_loc, c_f_pointer

use smartredis_dataset, only : dataset_type
use fortran_c_interop, only : convert_char_array_to_c

implicit none; private

#include "enums/enum_fortran.inc"
#include "client/client_interfaces.inc"
#include "client/put_tensor_interfaces.inc"
#include "client/unpack_tensor_interfaces.inc"
#include "client/misc_tensor_interfaces.inc"
#include "client/model_interfaces.inc"
#include "client/script_interfaces.inc"
#include "client/client_dataset_interfaces.inc"
#include "client/ensemble_interfaces.inc"

!> Stores all data and methods associated with the SmartRedis client that is used to communicate with the database
type, public :: client_type
  private

  logical(kind=c_bool) :: cluster = .false.        !< True if a database cluster is being used
  type(c_ptr)          :: client_ptr = c_null_ptr !< Pointer to the initialized SmartRedisClient
  logical              :: is_initialized = .false.    !< True if client is initialized
  contains

  ! Public procedures
  !> Puts a tensor into the database (overloaded)
  generic :: put_tensor => put_tensor_i8, put_tensor_i16, put_tensor_i32, put_tensor_i64, &
                           put_tensor_float, put_tensor_double
  !> Retrieve the tensor in the database into already allocated memory (overloaded)
  generic :: unpack_tensor => unpack_tensor_i8, unpack_tensor_i16, unpack_tensor_i32, unpack_tensor_i64, &
                              unpack_tensor_float, unpack_tensor_double

  !> Initializes a new instance of the SmartRedis client
  procedure :: initialize => initialize_client
  !> Check if a SmartRedis client has been initialized
  procedure :: isinitialized
  !> Destructs a new instance of the SmartRedis client
  procedure :: destructor
  !> Check the database for the existence of a specific model
  procedure :: model_exists
  !> Check the database for the existence of a specific tensor
  procedure :: tensor_exists
  !> Check the database for the existence of a specific key
  procedure :: key_exists
  !> Poll the database and return if the model exists
  procedure :: poll_model
  !> Poll the database and return if the tensor exists
  procedure :: poll_tensor
  !> Poll the database and return if the key exists
  procedure :: poll_key
  !> Rename a tensor within the database
  procedure :: rename_tensor
  !> Delete a tensor from the database
  procedure :: delete_tensor
  !> Copy a tensor within the database to a new key
  procedure :: copy_tensor
  !> Set a model from a file
  procedure :: set_model_from_file
  !> Set a model from a byte string that has been loaded within the application
  procedure :: set_model
  !> Retrieve the model as a byte string
  procedure :: get_model
  !> Set a script from a specified file
  procedure :: set_script_from_file
  !> Set a script as a byte or text string
  procedure :: set_script
  !> Retrieve the script from the database
  procedure :: get_script
  !> Run a script that has already been stored in the database
  procedure :: run_script
  !> Run a model that has already been stored in the database
  procedure :: run_model
  !> Put a SmartRedis dataset into the database
  procedure :: put_dataset
  !> Retrieve a SmartRedis dataset from the database
  procedure :: get_dataset
  !> Rename the dataset within the database
  procedure :: rename_dataset
  !> Copy a dataset stored in the database into another key
  procedure :: copy_dataset
  !> Delete the dataset from the database
  procedure :: delete_dataset

  procedure :: use_tensor_ensemble_prefix
  procedure :: use_model_ensemble_prefix
  procedure :: set_data_source


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

end type client_type

contains

!> Initializes a new instance of a SmartRedis client
subroutine initialize_client( self, cluster )
  class(client_type), intent(inout) :: self    !< Receives the initialized client
  logical, optional,  intent(in   ) :: cluster !< If true, client uses a database cluster (Default: .false.)

  if (present(cluster)) self%cluster = cluster
  self%client_ptr = c_constructor(self%cluster)
  self%is_initialized = .true. 
end subroutine initialize_client

logical function isinitialized(this)
  class(client_type) :: this
  isinitialized = this%is_initialized
end function isinitialized

!> A destructor for the SmartRedis client
subroutine destructor( self )
  class(client_type), intent(inout) :: self

  call c_destructor(self%client_ptr)
  self%client_ptr = C_NULL_PTR
end subroutine destructor

!> Check if the specified key exists in the database
logical function key_exists(self, key)
  class(client_type), intent(in) :: self !< The client
  character(len=*),   intent(in) :: key  !< The key to check

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: c_key_length

  c_key = trim(key)
  c_key_length = len_trim(key)

  key_exists = key_exists_c( self%client_ptr, c_key, c_key_length)
end function key_exists

!> Check if the specified model exists in the database
logical function model_exists(self, model_name)
  class(client_type), intent(in) :: self       !< The client
  character(len=*),   intent(in) :: model_name !< The model to check

  ! Local variables
  character(kind=c_char, len=len_trim(model_name)) :: c_model_name
  integer(kind=c_size_t) :: c_model_name_length

  c_model_name = trim(model_name)
  c_model_name_length = len_trim(model_name)

  model_exists = model_exists_c( self%client_ptr, c_model_name, c_model_name_length)
end function model_exists

!> Check if the specified tensor exists in the database
logical function tensor_exists(self, tensor_name)
  class(client_type), intent(in) :: self        !< The client
  character(len=*),   intent(in) :: tensor_name !< The tensor to check

  ! Local variables
  character(kind=c_char, len=len_trim(tensor_name)) :: c_tensor_name
  integer(kind=c_size_t) :: c_tensor_name_length

  c_tensor_name = trim(tensor_name)
  c_tensor_name_length = len_trim(tensor_name)

  tensor_exists = tensor_exists_c( self%client_ptr, c_tensor_name, c_tensor_name_length)
end function tensor_exists

!> Repeatedly poll the database until the tensor exists or the number of tries is exceeded
logical function poll_tensor( self, tensor_name, poll_frequency_ms, num_tries )
  class(client_type), intent(in) :: self              !< The client
  character(len=*),   intent(in) :: tensor_name       !< Key in the database to poll
  integer,            intent(in) :: poll_frequency_ms !< Frequency at which to poll the database (ms)
  integer,            intent(in) :: num_tries         !< Number of times to poll the database before failing

  ! Local variables
  character(kind=c_char,len=len_trim(tensor_name)) :: c_tensor_name
  integer(kind=c_size_t) :: c_tensor_name_length
  integer(kind=c_int) :: c_poll_frequency, c_num_tries

  c_tensor_name = trim(tensor_name)
  c_tensor_name_length = len_trim(tensor_name)
  c_num_tries = num_tries
  c_poll_frequency = poll_frequency_ms

  poll_tensor = poll_tensor_c(self%client_ptr, c_tensor_name, c_tensor_name_length, c_poll_frequency, c_num_tries)
end function poll_tensor

!> Repeatedly poll the database until the model exists or the number of tries is exceeded
logical function poll_model( self, model_name, poll_frequency_ms, num_tries )
  class(client_type), intent(in) :: self              !< The client
  character(len=*),   intent(in) :: model_name        !< Key in the database to poll
  integer,            intent(in) :: poll_frequency_ms !< Frequency at which to poll the database (ms)
  integer,            intent(in) :: num_tries         !< Number of times to poll the database before failing

  ! Local variables
  character(kind=c_char,len=len_trim(model_name)) :: c_model_name
  integer(kind=c_size_t) :: c_model_name_length
  integer(kind=c_int) :: c_poll_frequency, c_num_tries

  c_model_name = trim(model_name)
  c_model_name_length = len_trim(model_name)
  c_num_tries = num_tries
  c_poll_frequency = poll_frequency_ms

  poll_model = poll_model_c(self%client_ptr, c_model_name, c_model_name_length, c_poll_frequency, c_num_tries)
end function poll_model

!> Repeatedly poll the database until the key exists or the number of tries is exceeded
logical function poll_key( self, key, poll_frequency_ms, num_tries )
  class(client_type), intent(in) :: self              !< The client
  character(len=*),   intent(in) :: key               !< Key in the database to poll
  integer,            intent(in) :: poll_frequency_ms !< Frequency at which to poll the database (ms)
  integer,            intent(in) :: num_tries         !< Number of times to poll the database before failing

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: c_key_length
  integer(kind=c_int) :: c_poll_frequency, c_num_tries

  c_key = trim(key)
  c_key_length = len_trim(key)
  c_num_tries = num_tries
  c_poll_frequency = poll_frequency_ms

  poll_key = poll_key_c(self%client_ptr, c_key, c_key_length, c_poll_frequency, c_num_tries)
end function poll_key

!> Put a tensor whose Fortran type is the equivalent 'int8' C-type
subroutine put_tensor_i8(self, key, data, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'client/put_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int8
  call put_tensor_c(self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i8

!> Put a tensor whose Fortran type is the equivalent 'int16' C-type
subroutine put_tensor_i16(self, key, data, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'client/put_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int16
  call put_tensor_c(self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i16

!> Put a tensor whose Fortran type is the equivalent 'int32' C-type
subroutine put_tensor_i32(self, key, data, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'client/put_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int32
  call put_tensor_c(self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i32

!> Put a tensor whose Fortran type is the equivalent 'int64' C-type
subroutine put_tensor_i64(self, key, data, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'client/put_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int64
  call put_tensor_c(self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_i64

!> Put a tensor whose Fortran type is the equivalent 'float' C-type
subroutine put_tensor_float(self, key, data, dims)
  real(kind=c_float), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'client/put_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_flt
  call put_tensor_c(self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_float

!> Put a tensor whose Fortran type is the equivalent 'double' C-type
subroutine put_tensor_double(self, key, data, dims)
  real(kind=c_double), dimension(..), target, intent(in) :: data !< Data to be sent
  include 'client/put_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call put_tensor_c(self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, c_fortran_contiguous)
end subroutine put_tensor_double

!> Put a tensor whose Fortran type is the equivalent 'int8' C-type
subroutine unpack_tensor_i8(self, key, result, dims)
  integer(kind=c_int8_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'client/unpack_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int8
  call unpack_tensor_c( self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i8

!> Put a tensor whose Fortran type is the equivalent 'int16' C-type
subroutine unpack_tensor_i16(self, key, result, dims)
  integer(kind=c_int16_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'client/unpack_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int16
  call unpack_tensor_c( self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i16

!> Put a tensor whose Fortran type is the equivalent 'int32' C-type
subroutine unpack_tensor_i32(self, key, result, dims)
  integer(kind=c_int32_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'client/unpack_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int32
  call unpack_tensor_c( self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i32

!> Put a tensor whose Fortran type is the equivalent 'int64' C-type
subroutine unpack_tensor_i64(self, key, result, dims)
  integer(kind=c_int64_t), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'client/unpack_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_int64
  call unpack_tensor_c( self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_i64

!> Put a tensor whose Fortran type is the equivalent 'float' C-type
subroutine unpack_tensor_float(self, key, result, dims)
  real(kind=c_float), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'client/unpack_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_flt
  call unpack_tensor_c( self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_float

!> Put a tensor whose Fortran type is the equivalent 'double' C-type
subroutine unpack_tensor_double(self, key, result, dims)
  real(kind=c_double), dimension(..), target, intent(out) :: result !< Data to be sent
  include 'client/unpack_tensor_methods_common.inc'

  ! Define the type and call the C-interface
  data_type = tensor_dbl
  call unpack_tensor_c( self%client_ptr, c_key, key_length, data_ptr, c_dims_ptr, c_n_dims, data_type, mem_layout )
end subroutine unpack_tensor_double

!> Move a tensor to a new key
subroutine rename_tensor(self, key, new_key)
  class(client_type), intent(in) :: self    !< The initialized Fortran SmartRedis client
  character(len=*),   intent(in) :: key     !< The current key for the tensor
                                            !! excluding null terminating character
  character(len=*),   intent(in) :: new_key !< The new tensor key

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=len_trim(new_key)) :: c_new_key
  integer(kind=c_size_t) :: key_length, new_key_length

  c_key = trim(key)
  c_new_key = trim(new_key)

  key_length = len_trim(key)
  new_key_length = len_trim(new_key)

  call rename_tensor_c(self%client_ptr, c_key, key_length, c_new_key, new_key_length)
end subroutine rename_tensor

!> Delete a tensor
subroutine delete_tensor(self, key)
  class(client_type), intent(in) :: self !< The initialized Fortran SmartRedis client
  character(len=*),   intent(in) :: key  !< The key associated with the tensor

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: key_length

  c_key = trim(key)
  key_length = len_trim(key)

  call delete_tensor_c(self%client_ptr, c_key, key_length)
end subroutine delete_tensor

!> Copy a tensor to the destination key
subroutine copy_tensor(self, src_name, dest_name)
  class(client_type), intent(in) :: self      !< The initialized Fortran SmartRedis client
  character(len=*),   intent(in) :: src_name  !< The key associated with the tensor
                                              !! excluding null terminating character
  character(len=*),   intent(in) :: dest_name !< The new tensor key

  ! Local variables
  character(kind=c_char, len=len_trim(src_name)) :: c_src_name
  character(kind=c_char, len=len_trim(dest_name)) :: c_dest_name
  integer(kind=c_size_t) :: src_name_length, dest_name_length

  c_src_name = trim(src_name)
  c_dest_name = trim(dest_name)

  src_name_length = len_trim(src_name, kind=c_size_t)
  dest_name_length = len_trim(dest_name, kind=c_size_t)

  call copy_tensor_c(self%client_ptr, c_src_name, src_name_length, c_dest_name, dest_name_length)
end subroutine copy_tensor

!> Retrieve the model from the database
subroutine get_model(self, key, model)
  class(client_type),               intent(in   ) :: self  !< An initialized SmartRedis client
  character(len=*),                 intent(in   ) :: key   !< The key associated with the model
  character(len=*),                 intent(  out) :: model !< The model as a continuous buffer

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: key_length, model_length
  character(kind=c_char), dimension(:), pointer :: f_str_ptr
  type(c_ptr) :: c_str_ptr
  integer :: i

  c_key = trim(key)
  key_length = len_trim(key)

  c_str_ptr = get_model_c(self%client_ptr, key, key_length, c_str_ptr, model_length)

  call c_f_pointer(c_str_ptr, f_str_ptr, [ model_length ])

  do i=1,model_length
    model(i:i) = f_str_ptr(i)
  enddo

end subroutine get_model

!> Load the machine learning model from a file and set the configuration
subroutine set_model_from_file( self, key, model_file, backend, device, batch_size, min_batch_size, tag, &
    inputs, outputs )
  class(client_type),                       intent(in) :: self           !< An initialized SmartRedis client
  character(len=*),                         intent(in) :: key            !< The key to use to place the model
  character(len=*),                         intent(in) :: model_file     !< The file storing the model
  character(len=*),                         intent(in) :: backend        !< The name of the backend (TF, TFLITE, TORCH, ONNX)
  character(len=*),                         intent(in) :: device         !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  integer,                        optional, intent(in) :: batch_size     !< The batch size for model execution
  integer,                        optional, intent(in) :: min_batch_size !< The minimum batch size for model execution
  character(len=*),               optional, intent(in) :: tag            !< A tag to attach to the model for
                                                                         !! information purposes
  character(len=*), dimension(:), optional, intent(in) :: inputs         !< One or more names of model input nodes (TF
                                                                         !! models)
  character(len=*), dimension(:), optional, intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=len_trim(model_file)) :: c_model_file
  character(kind=c_char, len=len_trim(backend)) :: c_backend
  character(kind=c_char, len=len_trim(device)) :: c_device
  character(kind=c_char, len=:), allocatable :: c_tag

  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)
  character(kind=c_char,len=1), target, dimension(1) :: dummy_inputs, dummy_outputs

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: key_length, model_file_length, backend_length, device_length, tag_length, n_inputs, &
                            n_outputs
  integer(kind=c_int)    :: c_batch_size, c_min_batch_size
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  ! Set default values for the optional inputs
  c_batch_size = 0
  if (present(batch_size)) c_batch_size = batch_size
  c_min_batch_size = 0
  if (present(min_batch_size)) c_min_batch_size = min_batch_size
  if (present(tag)) then
    allocate( character(kind=c_char, len=len_trim(tag)) :: c_tag)
    c_tag = tag
    tag_length = len_trim(tag)
  else
    allocate( character(kind=c_char, len=1) :: c_tag)
    c_tag = ''
    tag_length = 1
  endif

  ! Cast to c_char kind stringts
  c_key = trim(key)
  c_model_file = trim(model_file)
  c_backend = trim(backend)
  c_device = trim(device)

  key_length = len_trim(key)
  model_file_length = len_trim(model_file)
  backend_length = len_trim(backend)
  device_length = len_trim(device)

  dummy_inputs = ''
  if (present(inputs)) then
    call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                  n_inputs)
  else
    call convert_char_array_to_c( dummy_inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr,&
                                  n_inputs)
  endif

  dummy_outputs =''
  if (present(outputs)) then
    call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, output_lengths_ptr,&
                                  n_outputs)
  else
    call convert_char_array_to_c( dummy_outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                  output_lengths_ptr, n_outputs)
  endif

  call set_model_from_file_c(self%client_ptr, c_key, key_length, c_model_file, model_file_length,               &
                             c_backend, backend_length, c_device, device_length, c_batch_size, c_min_batch_size,&
                             c_tag, tag_length, inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr,           &
                             output_lengths_ptr, n_outputs)
  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(ptrs_to_inputs)
  deallocate(c_outputs)
  deallocate(output_lengths)
  deallocate(ptrs_to_outputs)
  deallocate(c_tag)
end subroutine set_model_from_file

!> Establish a model to run
subroutine set_model( self, key, model, backend, device, batch_size, min_batch_size, tag, &
    inputs, outputs )
  class(client_type),             intent(in) :: self           !< An initialized SmartRedis client
  character(len=*),               intent(in) :: key            !< The key to use to place the model
  character(len=*),               intent(in) :: model          !< The binary representation of the model
  character(len=*),               intent(in) :: backend        !< The name of the backend (TF, TFLITE, TORCH, ONNX)
  character(len=*),               intent(in) :: device         !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  integer,                        intent(in) :: batch_size     !< The batch size for model execution
  integer,                        intent(in) :: min_batch_size !< The minimum batch size for model execution
  character(len=*),               intent(in) :: tag            !< A tag to attach to the model for information purposes
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of model input nodes (TF models)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=len_trim(model)) :: c_model
  character(kind=c_char, len=len_trim(backend)) :: c_backend
  character(kind=c_char, len=len_trim(device)) :: c_device
  character(kind=c_char, len=len_trim(tag)) :: c_tag

  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: key_length, model_length, backend_length, device_length, tag_length, n_inputs, &
                            n_outputs
  integer(kind=c_int)    :: c_batch_size, c_min_batch_size
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  c_key = trim(key)
  c_model = trim(model)
  c_backend = trim(backend)
  c_device = trim(device)
  c_tag = trim(tag)

  key_length = len_trim(key)
  model_length = len_trim(model)
  backend_length = len_trim(backend)
  device_length = len_trim(device)
  tag_length = len_trim(tag)

  ! Copy the input array into a c_char
  call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                n_inputs)
  call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                output_lengths_ptr, n_outputs)

  ! Cast the batch sizes to C integers
  c_batch_size = batch_size
  c_min_batch_size = min_batch_size

  call set_model_c(self%client_ptr, c_key, key_length, c_model, model_length, c_backend, backend_length, &
                 c_device, device_length, batch_size, min_batch_size, c_tag, tag_length,                 &
                 inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr, output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(ptrs_to_inputs)
  deallocate(c_outputs)
  deallocate(output_lengths)
  deallocate(ptrs_to_outputs)
end subroutine set_model

!> Execute a model
subroutine run_model(self, key, inputs, outputs)
  class(client_type),             intent(in) :: self    !< An initialized SmartRedis client
  character(len=*),               intent(in) :: key     !< The key to use to place the model
  character(len=*), dimension(:), intent(in) :: inputs  !< One or more names of model input nodes (TF models)
  character(len=*), dimension(:), intent(in) :: outputs !< One or more names of model output nodes (TF models)

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: n_inputs, n_outputs, key_length
  type(c_ptr) :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  c_key = trim(key)
  key_length = len_trim(key)

  call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                n_inputs)
  call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                output_lengths_ptr, n_outputs)

  call run_model_c(self%client_ptr, c_key, key_length, inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr, &
                  output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(ptrs_to_inputs)
  deallocate(c_outputs)
  deallocate(output_lengths)
  deallocate(ptrs_to_outputs)
end subroutine run_model

!> Retrieve the script from the database
subroutine get_script(self, key, script)
  class(client_type), intent(in   ) :: self   !< An initialized SmartRedis client
  character(len=*),   intent(in   ) :: key    !< The key to use to place the script
  character(len=*),   intent(  out) :: script !< The script as a continuous buffer

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: key_length, script_length
  character(kind=c_char), dimension(:), pointer :: f_str_ptr
  type(c_ptr) :: c_str_ptr
  integer :: i

  c_key = trim(key)
  key_length = len_trim(key)

  call get_script_c(self%client_ptr, key, key_length, c_str_ptr, script_length)

  call c_f_pointer(c_str_ptr, f_str_ptr, [ script_length ])

  do i=1,script_length
    script(i:i) = f_str_ptr(i)
  enddo

end subroutine get_script

subroutine set_script_from_file( self, key, device, script_file )
  class(client_type), intent(in) :: self        !< An initialized SmartRedis client
  character(len=*),   intent(in) :: key         !< The key to use to place the script
  character(len=*),   intent(in) :: device      !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  character(len=*),   intent(in) :: script_file !< The file storing the script

  ! Local variables
  character(kind=c_char, len=len_trim(key))         :: c_key
  character(kind=c_char, len=len_trim(device))      :: c_device
  character(kind=c_char, len=len_trim(script_file)) :: c_script_file

  integer(kind=c_size_t) :: key_length
  integer(kind=c_size_t) :: script_file_length
  integer(kind=c_size_t) :: device_length

  c_key = trim(key)
  c_script_file = trim(script_file)
  c_device = trim(device)

  key_length = len_trim(key)
  script_file_length = len_trim(script_file)
  device_length = len_trim(device)

  call set_script_from_file_c(self%client_ptr, c_key, key_length, c_device, device_length, &
                              c_script_file, script_file_length)

end subroutine set_script_from_file

subroutine set_script( self, key, device, script )
  class(client_type), intent(in) :: self   !< An initialized SmartRedis client
  character(len=*),   intent(in) :: key    !< The key to use to place the script
  character(len=*),   intent(in) :: device !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  character(len=*),   intent(in) :: script !< The file storing the script

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=len_trim(device)) :: c_device
  character(kind=c_char, len=len_trim(script)) :: c_script

  integer(kind=c_size_t) :: key_length
  integer(kind=c_size_t) :: script_length
  integer(kind=c_size_t) :: device_length

  c_key    = trim(key)
  c_script = trim(script)
  c_device = trim(device)

  key_length = len_trim(key)
  script_length = len_trim(script)
  device_length = len_trim(device)

  call set_script_c(self%client_ptr, c_key, key_length, c_device, device_length, c_script, script_length)

end subroutine set_script

subroutine run_script(self, key, func, inputs, outputs)
  class(client_type),             intent(in) :: self           !< An initialized SmartRedis client
  character(len=*),               intent(in) :: key            !< The key to use to place the script
  character(len=*),               intent(in) :: func           !< The name of the function in the script to call
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of script input nodes (TF scripts)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of script output nodes (TF scripts)

  ! Local variables
  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=len_trim(func)) :: c_func
  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: n_inputs, n_outputs, key_length, func_length
  type(c_ptr) :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  c_key  = trim(key)
  c_func = trim(func)

  key_length = len_trim(key)
  func_length = len_trim(func)

  call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                n_inputs)
  call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                output_lengths_ptr, n_outputs)

  call run_script_c(self%client_ptr, c_key, key_length, c_func, func_length, inputs_ptr, input_lengths_ptr, &
       n_inputs, outputs_ptr, output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(ptrs_to_inputs)
  deallocate(c_outputs)
  deallocate(output_lengths)
  deallocate(ptrs_to_outputs)

end subroutine run_script

!> Store a dataset in the database
subroutine put_dataset( self, dataset )
  class(client_type), intent(in) :: self    !< An initialized SmartRedis client
  type(dataset_type), intent(in) :: dataset !< Dataset to store in the dataset

  call put_dataset_c(self%client_ptr, dataset%dataset_ptr)
end subroutine put_dataset

!> Retrieve a dataset from the database
type(dataset_type) function get_dataset( self, name )
  class(client_type), intent(in) :: self !< An initialized SmartRedis client
  character(len=*),   intent(in) :: name !< Name of the dataset to get

  ! Local variables
  character(kind=c_char, len=len_trim(name)) :: c_name
  integer(kind=c_size_t) :: name_length

  c_name = trim(name)
  name_length = len_trim(name)
  get_dataset%dataset_ptr = get_dataset_c(self%client_ptr, c_name, name_length)
end function get_dataset

!> Rename a dataset stored in the database
subroutine rename_dataset( self, name, new_name )
  class(client_type), intent(in) :: self     !< An initialized SmartRedis client
  character(len=*),   intent(in) :: name     !< Original name of the dataset
  character(len=*),   intent(in) :: new_name !< New name of the dataset

  ! Local variables
  character(kind=c_char, len=len_trim(name)) :: c_name
  character(kind=c_char, len=len_trim(new_name)) :: c_new_name
  integer(kind=c_size_t) :: name_length, new_name_length

  c_name = trim(name)
  c_new_name = trim(new_name)
  name_length = len_trim(name)
  new_name_length = len_trim(new_name)

  call rename_dataset_c(self%client_ptr, c_name, name_length, c_new_name, new_name_length)
end subroutine rename_dataset

!> Copy a dataset within the database to a new name
subroutine copy_dataset( self, name, new_name )
  class(client_type), intent(in) :: self   !< An initialized SmartRedis client
  character(len=*),   intent(in) :: name     !< Source name of the dataset
  character(len=*),   intent(in) :: new_name !< Name of the new dataset

  ! Local variables
  character(kind=c_char, len=len_trim(name)) :: c_name
  character(kind=c_char, len=len_trim(new_name)) :: c_new_name
  integer(kind=c_size_t) :: name_length, new_name_length

  c_name = trim(name)
  c_new_name = trim(new_name)
  name_length = len_trim(name)
  new_name_length = len_trim(new_name)

  call copy_dataset_c(self%client_ptr, c_name, name_length, c_new_name, new_name_length)
end subroutine copy_dataset

!> Delete a dataset stored within a database
subroutine delete_dataset( self, name )
  class(client_type), intent(in) :: self !< An initialized SmartRedis client
  character(len=*),   intent(in) :: name   !< Name of the dataset to delete

  ! Local variables
  character(kind=c_char, len=len_trim(name)) :: c_name
  integer(kind=c_size_t) :: name_length

  c_name = trim(name)
  name_length = len_trim(name)
  call delete_dataset_c(self%client_ptr, c_name, name_length)
end subroutine delete_dataset

!> Set the data source (i.e. key prefix for get functions)
subroutine set_data_source( self, source_id )
  class(client_type), intent(in) :: self      !< An initialized SmartRedis client
  character(len=*),   intent(in) :: source_id !< The key prefix

  ! Local variables
  character(kind=c_char, len=len_trim(source_id)) :: c_source_id
  integer(kind=c_size_t) :: source_id_length
  c_source_id = trim(source_id)
  source_id_length = len_trim(source_id)

  call set_data_source_c( self%client_ptr, c_source_id, source_id_length )

end subroutine set_data_source

!> Set whether names of model and script entities should be prefixed (e.g. in an ensemble) to form database keys.
!! Prefixes will only be used if they were previously set through the environment variables SSKEYOUT and SSKEYIN.
!! Keys of entities created before client function is called will not be affected. By default, the client does not
!! prefix model and script keys.
subroutine use_model_ensemble_prefix( self, use_prefix )
  class(client_type),   intent(in) :: self       !< An initialized SmartRedis client
  logical,              intent(in) :: use_prefix !< The prefix setting

  call use_model_ensemble_prefix_c( self%client_ptr, logical(use_prefix,kind=c_bool) )
end subroutine use_model_ensemble_prefix


!> Set whether names of tensor and dataset entities should be prefixed (e.g. in an ensemble) to form database keys.
!! Prefixes will only be used if they were previously set through the environment variables SSKEYOUT and SSKEYIN.
!! Keys of entities created before client function is called will not be affected. By default, the client prefixes
!! tensor and dataset keys with the first prefix specified with the SSKEYIN and SSKEYOUT environment variables.
subroutine use_tensor_ensemble_prefix( self, use_prefix )
  class(client_type),   intent(in) :: self       !< An initialized SmartRedis client
  logical,              intent(in) :: use_prefix !< The prefix setting

  call use_tensor_ensemble_prefix_c( self%client_ptr, logical(use_prefix,kind=c_bool) )
end subroutine use_tensor_ensemble_prefix

end module smartredis_client
