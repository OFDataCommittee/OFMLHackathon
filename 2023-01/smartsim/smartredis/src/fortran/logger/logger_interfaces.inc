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
  subroutine c_log_data(context, level, data, data_length) bind(c, name="log_data_noexcept")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    type(c_ptr),             value :: context
    integer(kind=enum_kind), value :: level
    character(kind=c_char)         :: data(*)
    integer(kind=c_size_t), value  :: data_length
  end subroutine c_log_data
end interface

interface
  subroutine c_log_warning(context, level, data, data_length) bind(c, name="log_warning_noexcept")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    type(c_ptr),             value :: context
    integer(kind=enum_kind), value :: level
    character(kind=c_char)         :: data(*)
    integer(kind=c_size_t), value  :: data_length
  end subroutine c_log_warning
end interface

interface
  subroutine c_log_error(context, level, data, data_length) bind(c, name="log_error_noexcept")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    type(c_ptr),             value :: context
    integer(kind=enum_kind), value :: level
    character(kind=c_char)         :: data(*)
    integer(kind=c_size_t), value  :: data_length
  end subroutine c_log_error
end interface


interface
  subroutine c_log_data_string(context, context_length, level, data, data_length) &
    bind(c, name="log_data_noexcept_string")
    use iso_c_binding, only : c_char, c_size_t
    import :: enum_kind
    character(kind=c_char)         :: context(*)
    integer(kind=c_size_t), value  :: context_length
    integer(kind=enum_kind), value :: level
    character(kind=c_char)         :: data(*)
    integer(kind=c_size_t), value  :: data_length
  end subroutine c_log_data_string
end interface

interface
  subroutine c_log_warning_string(context, context_length, level, data, data_length) &
    bind(c, name="log_warning_noexcept_string")
    use iso_c_binding, only : c_char, c_size_t
    import :: enum_kind
    character(kind=c_char)         :: context(*)
    integer(kind=c_size_t), value  :: context_length
    integer(kind=enum_kind), value :: level
    character(kind=c_char)         :: data(*)
    integer(kind=c_size_t), value  :: data_length
  end subroutine c_log_warning_string
end interface

interface
  subroutine c_log_error_string(context, context_length, level, data, data_length) &
    bind(c, name="log_error_noexcept_string")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    character(kind=c_char)         :: context(*)
    integer(kind=c_size_t), value  :: context_length
    integer(kind=enum_kind), value :: level
    character(kind=c_char)         :: data(*)
    integer(kind=c_size_t), value  :: data_length
  end subroutine c_log_error_string
end interface
