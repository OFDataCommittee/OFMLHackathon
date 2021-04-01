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
  subroutine add_tensor_c(dataset, name, name_length, data, dims, n_dims, data_type, mem_layout) &
      bind(c, name="add_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    type(c_ptr),             value, intent(in) :: dataset      !< Pointer to the initialized client
    character(kind=c_char),         intent(in) :: name(*)     !< The name to use to place the tensor
    integer(kind=c_size_t),  value, intent(in) :: name_length !< The length of the name c-string,
                                                              !! excluding null terminating character
    type(c_ptr),             value, intent(in) :: data        !< A c ptr to the beginning of the data
    type(c_ptr),             value, intent(in) :: dims        !< Length along each dimension of the tensor
    integer(kind=c_size_t),  value, intent(in) :: n_dims      !< The number of dimensions of the tensor
    integer(kind=enum_kind), value, intent(in) :: data_type   !< The data type of the tensor
    integer(kind=enum_kind), value, intent(in) :: mem_layout  !< The memory layout of the data
  end subroutine add_tensor_c
end interface