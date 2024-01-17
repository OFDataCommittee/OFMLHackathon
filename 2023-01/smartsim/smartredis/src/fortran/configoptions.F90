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

module smartredis_configoptions

use iso_c_binding,   only : c_ptr, c_bool, c_null_ptr, c_char, c_int
use iso_c_binding,   only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t
use iso_c_binding,   only : c_loc, c_f_pointer

use, intrinsic :: iso_fortran_env, only: stderr => error_unit

use fortran_c_interop, only : convert_char_array_to_c, enum_kind, C_MAX_STRING

implicit none; private

include 'enum_fortran.inc'
include 'configoptions/configoptions_interfaces.inc'

public :: enum_kind !< The kind of integer equivalent to a C enum. According to C an Fortran
                    !! standards this should be c_int, but is renamed here to ensure that
                    !! users do not have to import the iso_c_binding module into their
                    !! programs

!> Contains multiple tensors and metadata used to describe an entire set of data
type, public :: configoptions_type
  type(c_ptr) :: configoptions_ptr !< A pointer to the initialized dataset object

  contains

  !> Access the raw C pointer for the configoptions
  procedure :: get_c_pointer

  ! Factory methods
  !> Instantiate ConfigOptions, getting selections from environment variables
  procedure :: create_configoptions_from_environment

  ! Option access
  !> Retrieve the value of a numeric configuration option
  procedure :: get_integer_option
  !> Retrieve the value of a string configuration option
  procedure :: get_string_option
  !> Check whether a configuration option is set
  procedure :: is_configured

  ! Option value overrides
  !> Override the value of a numeric configuration option
  procedure :: override_integer_option
  !> Override the value of a string configuration option
  procedure :: override_string_option

end type configoptions_type

contains


!> Access the raw C pointer for the ConfigOptions
function get_c_pointer(self)
  type(c_ptr)                           :: get_c_pointer
  class(configoptions_type), intent(in) :: self
  get_c_pointer = self%configoptions_ptr
end function get_c_pointer

!> Instantiate ConfigOptions, getting selections from environment variables
function create_configoptions_from_environment(self, db_suffix) result(code)
  class(configoptions_type), intent(inout) :: self        !< Receives the configoptions
  character(len=*),    intent(in)          :: db_suffix   !< Suffix to apply to environment
                                                          !! variables; empty string for none
  integer(kind=enum_kind)                  :: code !< Result of the operation

  ! Local variables
  integer(kind=c_size_t) :: db_suffix_length
  character(kind=c_char, len=len_trim(db_suffix)) :: c_db_suffix

  db_suffix_length = len_trim(db_suffix)
  c_db_suffix = trim(db_suffix)

  code = create_configoptions_from_environment_c( &
    c_db_suffix, db_suffix_length, self%configoptions_ptr)
end function create_configoptions_from_environment

!> Retrieve the value of a numeric configuration option
function get_integer_option(self, option_name, result) result(code)
  class(configoptions_type), intent(in) :: self          !< The configoptions
  character(len=*),          intent(in) :: option_name   !< The name of the configuration
                                                         !! option to retrieve
  integer(kind=c_int64_t),   intent(inout) :: result     !< Receives value of option
  integer(kind=enum_kind)               :: code

  ! Local variables
  character(kind=c_char, len=len_trim(option_name)) :: c_option_name
  integer(kind=c_size_t) :: c_option_name_length

  c_option_name = trim(option_name)
  c_option_name_length = len_trim(option_name)

  code = get_integer_option_c( &
    self%configoptions_ptr, c_option_name, c_option_name_length, result)
end function get_integer_option

!> Retrieve the value of a string configuration option
function get_string_option(self, option_name, result) result(code)
  class(configoptions_type), intent(in)  :: self          !< The configoptions
  character(len=*),          intent(in)  :: option_name   !< The name of the configuration
                                                          !! option to retrieve
  character(len=:), allocatable, intent(out) :: result    !< Receives value of option
  integer(kind=enum_kind)                :: code

  ! Local variables
  character(kind=c_char, len=len_trim(option_name)) :: c_option_name
  integer(kind=c_size_t) :: c_option_name_length
  integer(kind=c_size_t) :: c_result_length, i
  character(kind=c_char), dimension(:), pointer :: f_result_ptr
  type(c_ptr) :: c_result_ptr

  c_option_name = trim(option_name)
  c_option_name_length = len_trim(option_name)

  code = get_string_option_c( &
    self%configoptions_ptr, c_option_name, c_option_name_length, &
    c_result_ptr, c_result_length)

  ! Translate the string result if we got a valid one
  if (code .eq. SRNoError) then
    call c_f_pointer(c_result_ptr, f_result_ptr, [ c_result_length ])

    ALLOCATE(character(len=c_result_length) :: result)
    do i = 1, c_result_length
      result(i:i) = f_result_ptr(i)
    enddo
  endif
end function get_string_option

!> Check whether a configuration option is set
function is_configured(self, option_name, result) result(code)
  class(configoptions_type), intent(in)    :: self          !< The configoptions
  character(len=*),          intent(in)    :: option_name   !< The name of the configuration
                                                         !! option to check
  logical,                   intent(inout) :: result     !< Receives value of option
  integer(kind=enum_kind)                  :: code

  ! Local variables
  character(kind=c_char, len=len_trim(option_name)) :: c_option_name
  integer(kind=c_size_t) :: c_option_name_length
  logical(kind=c_bool) :: c_result

  c_option_name = trim(option_name)
  c_option_name_length = len_trim(option_name)

  code = is_configured_c( &
    self%configoptions_ptr, c_option_name, c_option_name_length, c_result)
    result = c_result
end function is_configured

!> Override the value of a numeric configuration option
function override_integer_option(self, option_name, value) result(code)
  class(configoptions_type), intent(in) :: self        !< The configoptions
  character(len=*),          intent(in) :: option_name !< The name of the configuration
                                                       !! option to override
  integer(kind=c_int64_t),   intent(in) :: value       !< The value to store for the option
  integer(kind=enum_kind)               :: code

  ! Local variables
  character(kind=c_char, len=len_trim(option_name)) :: c_option_name
  integer(kind=c_size_t) :: c_option_name_length

  c_option_name = trim(option_name)
  c_option_name_length = len_trim(option_name)

  code = override_integer_option_c( &
    self%configoptions_ptr, c_option_name, c_option_name_length, value)
end function override_integer_option

!> Override the value of a string configuration option
function override_string_option(self, option_name, value) result(code)
  class(configoptions_type), intent(in)  :: self        !< The configoptions
  character(len=*),          intent(in)  :: option_name !< The name of the configuration
                                                        !! option to override
  character(len=*),          intent(in)  :: value       !< The value to store for the option
  integer(kind=enum_kind)                :: code

  ! Local variables
  character(kind=c_char, len=len_trim(option_name)) :: c_option_name
  character(kind=c_char, len=len_trim(value)) :: c_value
  integer(kind=c_size_t) :: c_option_name_length, c_value_length

  c_option_name = trim(option_name)
  c_option_name_length = len_trim(option_name)
  c_value = trim(value)
  c_value_length = len_trim(value)

  code = override_string_option_c( &
    self%configoptions_ptr, c_option_name, c_option_name_length, &
    c_value, c_value_length)
end function override_string_option

end module smartredis_configoptions
