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
  use smartredis_configoptions, only : configoptions_type
  use test_utils,  only : setenv, unsetenv
  use iso_fortran_env, only : STDERR => error_unit
  use iso_c_binding, only : c_ptr, c_bool, c_null_ptr, c_char, c_int
  use iso_c_binding, only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t

  implicit none

#include "enum_fortran.inc"

  type(configoptions_type)      :: co
  integer                       :: result
  integer(kind=8)               :: ivalue, iresult ! int
  logical                       :: bvalue, bresult ! bool
  character(kind=c_char, len=:), allocatable :: svalue, sresult ! string

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! Establish test keys
  ! non-suffixed testing keys
  call setenv("test_integer_key", "42")
  call setenv("test_string_key", "charizard")
  ! suffixed testing keys
  call setenv("integer_key_suffixtest", "42")
  call setenv("string_key_suffixtest", "charizard")

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! non-suffixed option testing
  result = co%create_configoptions_from_environment("");
  if (result .ne. SRNoError) error stop

  ! integer option tests
  write(*,*) "ConfigOption testing: integer option tests"
  result = co%get_integer_option("test_integer_key", iresult)
  if (result .ne. SRNoError) error stop
  if (iresult .ne. 42) error stop

  result = co%is_configured( &
    "test_integer_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .eqv. .true.) error stop

  result = co%get_integer_option( &
    "test_integer_key_that_is_not_really_present", iresult)
  if (result .eq. SRNoError) error stop

  ivalue = 42
  result = co%override_integer_option( &
    "test_integer_key_that_is_not_really_present", ivalue)
  if (result .ne. SRNoError) error stop

  result = co%get_integer_option( &
    "test_integer_key_that_is_not_really_present", iresult)
  if (result .ne. SRNoError) error stop
  if (iresult .ne. 42) error stop

  result = co%is_configured( &
    "test_integer_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .neqv. .true.) error stop


  ! string option tests
  write(*,*) "ConfigOption testing: string option tests"
  result = co%get_string_option("test_string_key", sresult)
  if (result .ne. SRNoError) error stop
  if (sresult .ne. "charizard") error stop

  result = co%is_configured( &
    "test_string_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .eqv. .true.) error stop

  result = co%get_string_option( &
    "test_string_key_that_is_not_really_present", sresult)
  if (result .eq. SRNoError) error stop

  svalue = "meowth"
  result = co%override_string_option( &
    "test_string_key_that_is_not_really_present", svalue)
  if (result .ne. SRNoError) error stop

  result = co%get_string_option( &
    "test_string_key_that_is_not_really_present", sresult)
  if (result .ne. SRNoError) error stop
  if (sresult .ne. "meowth") error stop

  result = co%is_configured( &
    "test_string_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .neqv. .true.) error stop


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! suffixtest testing
  result = co%create_configoptions_from_environment("suffixtest");
  if (result .ne. SRNoError) error stop

  ! integer option tests
  write(*,*) "ConfigOption testing: suffixed integer option tests"

  result = co%get_integer_option("integer_key", iresult)
  if (result .ne. SRNoError) error stop
  if (iresult .ne. 42) error stop

  result = co%is_configured( &
    "integer_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .eqv. .true.) error stop

  result = co%get_integer_option( &
    "integer_key_that_is_not_really_present", iresult)
  if (result .eq. SRNoError) error stop

  ivalue = 42
  result = co%override_integer_option( &
    "integer_key_that_is_not_really_present", ivalue)
  if (result .ne. SRNoError) error stop

  result = co%get_integer_option( &
    "integer_key_that_is_not_really_present", iresult)
  if (result .ne. SRNoError) error stop
  if (iresult .ne. 42) error stop

  result = co%is_configured( &
    "integer_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .neqv. .true.) error stop


  ! string option tests
  write(*,*) "ConfigOption testing: suffixed string option tests"

  result = co%get_string_option("string_key", sresult)
  if (result .ne. SRNoError) error stop
  if (sresult .ne. "charizard") error stop

  result = co%is_configured("string_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .eqv. .true.) error stop

  result = co%get_string_option( &
    "string_key_that_is_not_really_present", sresult)
  if (result .eq. SRNoError) error stop

  svalue = "meowth"
  result = co%override_string_option( &
    "string_key_that_is_not_really_present", svalue)
  if (result .ne. SRNoError) error stop

  result = co%get_string_option( &
    "string_key_that_is_not_really_present", sresult)
  if (result .ne. SRNoError) error stop
  if (sresult .ne. "meowth") error stop

  result = co%is_configured("string_key_that_is_not_really_present", bresult)
  if (result .ne. SRNoError) error stop
  if (bresult .neqv. .true.) error stop


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! Clean up test keys
  ! non-prefixed testing keys
  call unsetenv("test_integer_key")
  call unsetenv("test_string_key")
  ! suffixed testing keys
  call unsetenv("integer_key_suffixtest")
  call unsetenv("string_key_suffixtest")

  ! Done
  write(*,*) "ConfigOption testing: passed"
end program main
