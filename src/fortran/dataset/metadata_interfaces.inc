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

interface
  subroutine add_meta_scalar_c( dataset, name, name_length, data, data_type ) bind(c, name="add_meta_scalar")
    use iso_c_binding, only : c_ptr, c_size_t, c_char
    import :: enum_kind
    type(c_ptr),             value, intent(in) :: dataset     !< A c_ptr to the dataset object
    character(kind=c_char),         intent(in) :: name(*)     !< The name to use to place the tensor
    integer(kind=c_size_t),  value, intent(in) :: name_length !< The length of the name c-string,
                                                              !! excluding null terminating character
    type(c_ptr),             value, intent(in) :: data        !< A c ptr to the beginning of the data
    integer(kind=enum_kind), value, intent(in) :: data_type   !< The c-type of the data
  end subroutine add_meta_scalar_c
end interface

interface
  subroutine add_meta_string_c( dataset, name, name_length, data, data_length ) bind(c, name="add_meta_string")
    use iso_c_binding, only : c_ptr, c_size_t, c_char
    import :: enum_kind
    type(c_ptr),             value, intent(in) :: dataset     !< A c_ptr to the dataset object
    character(kind=c_char),         intent(in) :: name(*)     !< The name to use to place the tensor
    integer(kind=c_size_t),  value, intent(in) :: name_length !< The length of the name c-string,
                                                              !! excluding null terminating character
    character(kind=c_char),         intent(in) :: data(*)     !< The actual metadata string
    integer(kind=c_size_t),  value, intent(in) :: data_length !< Length of the metadata string
  end subroutine add_meta_string_c
end interface

interface
  type(c_ptr) function get_meta_scalars_c( dataset, name, name_length, length, data_type ) &
      bind(c, name="get_meta_scalars")
    use iso_c_binding, only : c_ptr, c_size_t, c_char
    import :: enum_kind
    type(c_ptr),            value, intent(in) :: dataset      !< A c_ptr to the dataset object
    character(kind=c_char),        intent(in) :: name(*)      !< The name to use to reference the metadata
    integer(kind=c_size_t), value, intent(in) :: name_length  !< The length of the name c-string,
                                                              !! excluding null terminating character
    integer(kind=c_size_t),        intent(  out) :: length    !< The number of entries in the field
    integer(kind=enum_kind),       intent(  out) :: data_type !< The c-type of the data
  end function get_meta_scalars_c
end interface