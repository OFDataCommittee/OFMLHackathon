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

program main
  use smartredis_client, only : client_type
  use smartredis_dataset, only : dataset_type
  use smartredis_logcontext, only : logcontext_type
  use smartredis_logger, only : log_data, log_warning, log_error
  use test_utils, only : use_cluster
  use iso_fortran_env, only : STDERR => error_unit
  use iso_c_binding, only : c_ptr, c_bool, c_null_ptr, c_char, c_int
  use iso_c_binding, only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t

  implicit none

#include "enum_fortran.inc"

  type(client_type)     :: client
  type(dataset_type)    :: dataset
  type(logcontext_type) :: logcontext
  character(kind=c_char, len=:), allocatable :: string_context
  integer               :: result

  result = logcontext%initialize("client_test_logging (logcontext)")
  if (result .ne. SRNoError) error stop
  result = client%initialize("client_test_logging (client)")
  if (result .ne. SRNoError) error stop
  result = dataset%initialize("client_test_logging (dataset)")
  if (result .ne. SRNoError) error stop
  string_context = 'client_test_logging (string)'

  ! Logging against the Client
  ! ==========================
  call log_data(client, LLQuiet, &
    "This is data logged against the Client at the Quiet level")
  call log_data(client, LLInfo, &
    "This is data logged against the Client at the Info level")
  call log_data(client, LLDebug, &
    "This is data logged against the Client at the Debug level")
  call log_data(client, LLDeveloper, &
    "This is data logged against the Client at the Developer level")

  call log_warning(client, LLQuiet, &
    "This is a warning logged against the Client at the Quiet level")
  call log_warning(client, LLInfo, &
    "This is a warning logged against the Client at the Info level")
  call log_warning(client, LLDebug, &
    "This is a warning logged against the Client at the Debug level")
  call log_warning(client, LLDeveloper, &
    "This is a warning logged against the Client at the Developer level")

  call log_error(client, LLQuiet, &
    "This is an error logged against the Client at the Quiet level")
  call log_error(client, LLInfo, &
    "This is an error logged against the Client at the Info level")
  call log_error(client, LLDebug, &
    "This is an error logged against the Client at the Debug level")
  call log_error(client, LLDeveloper, &
    "This is an error logged against the Client at the Developer level")

  ! Logging against the Dataset
  ! ===========================
  call log_data(dataset, LLQuiet, &
    "This is data logged against the Dataset at the Quiet level")
  call log_data(dataset, LLInfo, &
    "This is data logged against the Dataset at the Info level")
  call log_data(dataset, LLDebug, &
    "This is data logged against the Dataset at the Debug level")
  call log_data(dataset, LLDeveloper, &
    "This is data logged against the Dataset at the Developer level")

  call log_warning(dataset, LLQuiet, &
    "This is a warning logged against the Dataset at the Quiet level")
  call log_warning(dataset, LLInfo, &
    "This is a warning logged against the Dataset at the Info level")
  call log_warning(dataset, LLDebug, &
    "This is a warning logged against the Dataset at the Debug level")
  call log_warning(dataset, LLDeveloper, &
    "This is a warning logged against the Dataset at the Developer level")

  call log_error(dataset, LLQuiet, &
    "This is an error logged against the Dataset at the Quiet level")
  call log_error(dataset, LLInfo, &
    "This is an error logged against the Dataset at the Info level")
  call log_error(dataset, LLDebug, &
    "This is an error logged against the Dataset at the Debug level")
  call log_error(dataset, LLDeveloper, &
    "This is an error logged against the Dataset at the Developer level")

  ! Logging against the LogContext
  ! ==============================
  call log_data(logcontext, LLQuiet, &
    "This is data logged against the LogContext at the Quiet level")
  call log_data(logcontext, LLInfo, &
    "This is data logged against the LogContext at the Info level")
  call log_data(logcontext, LLDebug, &
    "This is data logged against the LogContext at the Debug level")
  call log_data(logcontext, LLDeveloper, &
    "This is data logged against the LogContext at the Developer level")

  call log_warning(logcontext, LLQuiet, &
    "This is a warning logged against the LogContext at the Quiet level")
  call log_warning(logcontext, LLInfo, &
    "This is a warning logged against the LogContext at the Info level")
  call log_warning(logcontext, LLDebug, &
    "This is a warning logged against the LogContext at the Debug level")
  call log_warning(logcontext, LLDeveloper, &
    "This is a warning logged against the LogContext at the Developer level")

  call log_error(logcontext, LLQuiet, &
    "This is an error logged against the LogContext at the Quiet level")
  call log_error(logcontext, LLInfo, &
    "This is an error logged against the LogContext at the Info level")
  call log_error(logcontext, LLDebug, &
    "This is an error logged against the LogContext at the Debug level")
  call log_error(logcontext, LLDeveloper, &
    "This is an error logged against the LogContext at the Developer level")

  ! Logging against the string
  ! ==============================
  call log_data(string_context, LLQuiet, &
    "This is data logged against a string context at the Quiet level")
  call log_data(string_context, LLInfo, &
    "This is data logged against a string context at the Info level")
  call log_data(string_context, LLDebug, &
    "This is data logged against a string context at the Debug level")
  call log_data(string_context, LLDeveloper, &
    "This is data logged against a string context at the Developer level")

  call log_warning(string_context, LLQuiet, &
    "This is a warning logged against a string context at the Quiet level")
  call log_warning(string_context, LLInfo, &
    "This is a warning logged against a string context at the Info level")
  call log_warning(string_context, LLDebug, &
    "This is a warning logged against a string context at the Debug level")
  call log_warning(string_context, LLDeveloper, &
    "This is a warning logged against a string context at the Developer level")

  call log_error(string_context, LLQuiet, &
    "This is an error logged against a string context at the Quiet level")
  call log_error(string_context, LLInfo, &
    "This is an error logged against a string context at the Info level")
  call log_error(string_context, LLDebug, &
    "This is an error logged against a string context at the Debug level")
  call log_error(string_context, LLDeveloper, &
    "This is an error logged against a string context at the Developer level")

  ! Done
  write(*,*) "client logging: passed"
end program main
