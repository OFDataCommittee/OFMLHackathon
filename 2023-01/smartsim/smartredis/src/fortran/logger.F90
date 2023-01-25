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

module smartredis_logger

use iso_c_binding, only : c_ptr, c_char, c_size_t
use iso_c_binding, only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double
use iso_c_binding, only : c_loc, c_f_pointer
use fortran_c_interop, only : convert_char_array_to_c, enum_kind, C_MAX_STRING

use, intrinsic :: iso_fortran_env, only: stderr => error_unit
use smartredis_client, only : client_type
use smartredis_dataset, only : dataset_type
use smartredis_logcontext, only : logcontext_type

implicit none; private

#include "enum_fortran.inc"
#include "logger/logger_interfaces.inc"

interface log_data
  module procedure log_data_client, log_data_dataset, &
                   log_data_logcontext, log_data_string
end interface log_data

interface log_warning
  module procedure log_warning_client, log_warning_dataset, &
                   log_warning_logcontext, log_warning_string
end interface log_warning

interface log_error
  module procedure log_error_client, log_error_dataset, &
                   log_error_logcontext, log_error_string
end interface log_error

public :: enum_kind !< The kind of integer equivalent to a C enum. According to C an Fortran
                    !! standards this should be c_int, but is renamed here to ensure that
                    !! users do not have to import the iso_c_binding module into their
                    !! programs

public :: log_data, log_warning, log_error

contains

! log_data overloads
! ==================

!> Log data to the SmartRedis log against a Client object
subroutine log_data_client(context, level, data)
  type(client_type),        intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_data(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_data_client

!> Log data to the SmartRedis log against a Dataset object
subroutine log_data_dataset(context, level, data)
  type(dataset_type),       intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_data(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_data_dataset

!> Log data to the SmartRedis log against a Logcontext object
subroutine log_data_logcontext(context, level, data)
  type(logcontext_type),    intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_data(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_data_logcontext

!> Log data to the SmartRedis log against a string object
subroutine log_data_string(context, level, data)
  character(len=*),         intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_context, c_data
  integer(kind=c_size_t) :: c_context_length, c_data_length

  c_context = trim(context)
  c_context_length = len_trim(context)

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_data_string(&
    c_context, c_context_length, level, c_data, c_data_length)
  deallocate(c_data)
  deallocate(c_context)
end subroutine log_data_string

! log_warning overloads
! =====================

!> Log a warning to the SmartRedis log against a Client object
subroutine log_warning_client(context, level, data)
  type(client_type),        intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_warning(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_warning_client

!> Log a warning to the SmartRedis log against a Dataset object
subroutine log_warning_dataset(context, level, data)
  type(dataset_type),       intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_warning(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_warning_dataset

!> Log a warning to the SmartRedis log against a LogContext object
subroutine log_warning_logcontext(context, level, data)
  type(logcontext_type),    intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_warning(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_warning_logcontext

!> Log a warning to the SmartRedis log against a string object
subroutine log_warning_string(context, level, data)
  character(len=*),         intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_context, c_data
  integer(kind=c_size_t) :: c_context_length, c_data_length

  c_context = trim(context)
  c_context_length = len_trim(context)

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_warning_string(&
    c_context, c_context_length, level, c_data, c_data_length)
  deallocate(c_context)
  deallocate(c_data)
end subroutine log_warning_string

! log_error overloads
! ===================

!> Log an error to the SmartRedis log against a Client object
subroutine log_error_client(context, level, data)
  type(client_type),        intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_error(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_error_client

!> Log an error to the SmartRedis log against a Dataset object
subroutine log_error_dataset(context, level, data)
  type(dataset_type),       intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_error(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_error_dataset

!> Log an error to the SmartRedis log against a LogContext object
subroutine log_error_logcontext(context, level, data)
  type(logcontext_type),    intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_data
  integer(kind=c_size_t) :: c_data_length

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_error(context%get_c_pointer(), level, c_data, c_data_length)
  deallocate(c_data)
end subroutine log_error_logcontext

!> Log an error to the SmartRedis log against a string object
subroutine log_error_string(context, level, data)
  character(len=*),         intent(in) :: context !< Context for logging
  integer (kind=enum_kind), intent(in) :: level   !< Minimum logging level to log the data at
  character(len=*),         intent(in) :: data    !< Data to be logged

  ! Local variables
  character(kind=c_char, len=:), allocatable :: c_context, c_data
  integer(kind=c_size_t) :: c_context_length, c_data_length

  c_context = trim(context)
  c_context_length = len_trim(context)

  c_data = trim(data)
  c_data_length = len_trim(data)

  call c_log_error_string(&
    c_context, c_context_length, level, c_data, c_data_length)
  deallocate(c_context)
  deallocate(c_data)
end subroutine log_error_string

end module smartredis_logger
