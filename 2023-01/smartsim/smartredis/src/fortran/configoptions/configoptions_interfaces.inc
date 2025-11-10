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

interface

  function create_configoptions_from_environment_c( &
    db_suffix, db_suffix_len, configoptions) &
    bind(c, name="create_configoptions_from_environment")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: create_configoptions_from_environment_c
    character(kind=c_char)        :: db_suffix(*)  !< Suffix to add to environment vars
    integer(kind=c_size_t), value :: db_suffix_len !< How many characters in db_suffix
    type(c_ptr)                   :: configoptions !< Receives the constructed configoptions
  end function create_configoptions_from_environment_c

  function get_integer_option_c(cfgopts, option_name, option_name_len, result) &
    bind(c, name="get_integer_option")
    use iso_c_binding, only : c_ptr, c_char, c_size_t, c_int64_t
    import :: enum_kind
    type(c_ptr), value             :: cfgopts
    integer(kind=enum_kind)        :: get_integer_option_c
    character(kind=c_char)         :: option_name(*)    !< The name of the configuration
                                                        !! option to retrieve
    integer(kind=c_size_t), value  :: option_name_len   !< The length of the option_name string
    integer(kind=c_int64_t)        :: result            !< Receives the integer result
  end function get_integer_option_c

  function get_string_option_c(cfgopts, option_name, option_name_len, result, result_len) &
    bind(c, name="get_string_option")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    type(c_ptr), value            :: cfgopts
    integer(kind=enum_kind)       :: get_string_option_c
    character(kind=c_char)        :: option_name(*)    !< The name of the configuration
                                                       !! option to retrieve
    integer(kind=c_size_t), value :: option_name_len   !< The length of the option_name string
    type(c_ptr)                   :: result            !< Receives the string result
    integer(kind=c_size_t)        :: result_len        !< The length of the result string
  end function get_string_option_c

  function is_configured_c(cfgopts, option_name, option_name_len, result) &
    bind(c, name="is_configured")
    use iso_c_binding, only : c_ptr, c_char, c_size_t, c_bool
    import :: enum_kind
    type(c_ptr), value            :: cfgopts
    integer(kind=enum_kind)       :: is_configured_c
    character(kind=c_char)        :: option_name(*)    !< The name of the configuration
                                                       !! option to check
    integer(kind=c_size_t), value :: option_name_len   !< The length of the option_name string
    logical(kind=c_bool)          :: result            !< Receives the boolean result
  end function is_configured_c

  function override_integer_option_c(cfgopts, option_name, option_name_len, value) &
    bind(c, name="override_integer_option")
    use iso_c_binding, only : c_ptr, c_char, c_size_t, c_int64_t
    import :: enum_kind
    type(c_ptr), value             :: cfgopts
    integer(kind=enum_kind)        :: override_integer_option_c
    character(kind=c_char)         :: option_name(*)   !< The name of the configuration
                                                       !! option to override
    integer(kind=c_size_t), value  :: option_name_len  !< The length of the option_name string
    integer(kind=c_int64_t), value :: value            !< The value to store for the option
  end function override_integer_option_c

  function override_string_option_c(cfgopts, option_name, option_name_len, value, value_len) &
    bind(c, name="override_string_option")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    type(c_ptr), value            :: cfgopts
    integer(kind=enum_kind)       :: override_string_option_c
    character(kind=c_char)        :: option_name(*)    !< The name of the configuration
                                                       !! option to override
    integer(kind=c_size_t), value :: option_name_len   !< The length of the option_name string
    character(kind=c_char)        :: value(*)          !< The value to store for the option
    integer(kind=c_size_t), value :: value_len         !< The length of the value string
  end function override_string_option_c

end interface
