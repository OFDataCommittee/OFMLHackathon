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

  class(client_type),                   intent(in) :: self  !< Pointer to the initialized client
  character(len=*),                     intent(in) :: key   !< The key to use to place the tensor
  integer, dimension(:),                intent(in) :: dims  !< Length along each dimension of the tensor

  ! Local variables
  integer(kind=c_size_t) :: ndim ! Number of dimensions
  type(c_ptr) :: data_ptr, c_dims_ptr
  character(kind=c_char, len=len_trim(key)) :: c_key !< Transformed fortran 'key' to a c-string
  integer(kind=c_size_t) :: key_length, c_n_dims
  integer(kind=c_size_t), target :: c_dims(size(dims))
  integer(kind=enum_kind) :: data_type, mem_layout

  c_key = trim(key)
  key_length = len_trim(key)
  data_ptr = c_loc(result)
  c_dims(:) = dims(:)
  c_n_dims = size(dims)
  c_dims_ptr = c_loc(c_dims)
  mem_layout = c_fortran_contiguous
