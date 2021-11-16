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

! Dummy enum, used to set the integer kind used
enum, bind(c)
  enumerator :: dummy = 0
end enum
integer, parameter :: enum_kind = kind(dummy)

! The following enums have analogues to the C and C++ clients. Fortran does not haven named enums, but the name:value
! parameters must be the same. By convention negative enums, represent items that are not supported in the Fortran
! client.

! TensorType
enum, bind(c)
  enumerator :: tensor_invalid = -1
  enumerator :: tensor_dbl     = 1
  enumerator :: tensor_flt     = 2
  enumerator :: tensor_int8    = 3
  enumerator :: tensor_int16   = 4
  enumerator :: tensor_int32   = 5
  enumerator :: tensor_int64   = 6
  enumerator :: tensor_uint8   = -1
  enumerator :: tensor_uint16  = -1
end enum

! MemoryLayout​
enum, bind(c)
  enumerator :: c_invalid            = -1
  enumerator :: c_nested             = 1
  enumerator :: c_contiguous         = 2
  enumerator :: c_fortran_nested     = 3
  enumerator :: c_fortran_contiguous = 4
end enum

! MetadataType​
enum, bind(c)
  enumerator :: meta_invalid = -1
  enumerator :: meta_dbl     = 1
  enumerator :: meta_flt     = 2
  enumerator :: meta_int32   = 3
  enumerator :: meta_int64   = 4
  enumerator :: meta_uint32  = -1
  enumerator :: meta_uint64  = -1
  enumerator :: meta_string  = 7
end enum

! SRError
enum, bind(c)
  enumerator :: sr_ok        = 0
  enumerator :: sr_badalloc  = 1
  enumerator :: sr_dberr     = 2
  enumerator :: sr_internal  = 3
  enumerator :: sr_runtime   = 4
  enumerator :: sr_parameter = 5
  enumerator :: sr_timeout   = 6
  enumerator :: sr_invalid   = -1
end enum
