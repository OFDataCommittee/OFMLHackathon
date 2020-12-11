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
  enumerator :: tensor_dbl    = 1
  enumerator :: tensor_flt    = 2
  enumerator :: tensor_int8   = 3
  enumerator :: tensor_int16  = 4
  enumerator :: tensor_int32  = 5
  enumerator :: tensor_int64  = 6
  enumerator :: tensor_uint8  = -1
  enumerator :: tensor_uint16 = -1
end enum

! MemoryLayout​
enum, bind(c)
  enumerator :: c_nested             = 1
  enumerator :: c_contiguous         = 2
  enumerator :: c_fortran_nested     = 3
  enumerator :: c_fortran_contiguous = 4
end enum

! MetadataType​
enum, bind(c)
  enumerator :: meta_dbl     = 1
  enumerator :: meta_flt     = 2
  enumerator :: meta_int32   = 3
  enumerator :: meta_int64   = 4
  enumerator :: meta_uint32  = -1
  enumerator :: meta_uint64  = -1
  enumerator :: meta_string  = 7
end enum
