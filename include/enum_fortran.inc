! BSD 2-Clause License
!
! Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
  enumerator :: tensor_invalid = -1 ! Invalid or uninitialized tensor type
  enumerator :: tensor_dbl     = 1  ! Double-precision floating point tensor type
  enumerator :: tensor_flt     = 2  ! Floating point tensor type
  enumerator :: tensor_int8    = 3  ! 8-bit signed integer tensor type
  enumerator :: tensor_int16   = 4  ! 16-bit signed integer tensor type
  enumerator :: tensor_int32   = 5  ! 32-bit signed integer tensor type
  enumerator :: tensor_int64   = 6  ! 64-bit signed integer tensor type
  enumerator :: tensor_uint8   = -1 ! 8-bit unsigned integer tensor type
  enumerator :: tensor_uint16  = -1 ! 16-bit unsigned integer tensor type
end enum

! MemoryLayout​
enum, bind(c)
  enumerator :: c_invalid            = -1 ! Invalid or uninitialized memory layout
  enumerator :: c_nested             = 1  ! Multidimensional row-major array layout with nested arrays of
                                          ! pointers (contiguous at innermost layer)
  enumerator :: c_contiguous         = 2  ! Multidimensional row-major array layout in contiguous memory
  enumerator :: c_fortran_nested     = 3  ! Multidimensional column-major array layout with nested arrays of
                                          ! pointers (contiguous at innermost layer)
  enumerator :: c_fortran_contiguous = 4  ! Multidimensional column-major array layout in contiguous memory
end enum

! MetadataType​
enum, bind(c)
  enumerator :: meta_invalid = -1 ! Invalid or uninitialized metadata
  enumerator :: meta_dbl     = 1  ! Double-precision floating point metadata
  enumerator :: meta_flt     = 2  ! Floating point metadata
  enumerator :: meta_int32   = 3  ! 32-bit signed integer metadata
  enumerator :: meta_int64   = 4  ! 64-bit signed integer metadata
  enumerator :: meta_uint32  = -1 ! 32-bit unsigned integer metadata
  enumerator :: meta_uint64  = -1 ! 64-bit unsigned integer metadata
  enumerator :: meta_string  = 7  ! ASCII text string metadata
end enum

! SRError
enum, bind(c)
  enumerator :: SRNoError        = 0  ! No error
  enumerator :: SRBadAllocError  = 1  ! Memory allocation error
  enumerator :: SRDatabaseError  = 2  ! Backend database error
  enumerator :: SRInternalError  = 3  ! Internal SmartRedis error
  enumerator :: SRRuntimeError   = 4  ! Runtime error executing an operation
  enumerator :: SRParameterError = 5  ! Bad parameter error
  enumerator :: SRTimeoutError   = 6  ! Timeout error
  enumerator :: SRKeyError       = 7  ! Key error
  enumerator :: SRInvalidError   = -1 ! Uninitialized error variable
  enumerator :: SRTypeError      = 9  ! Type mismatch
end enum
