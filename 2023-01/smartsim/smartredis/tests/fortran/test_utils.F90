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

module test_utils
  use iso_c_binding, only : c_char, c_int, c_null_char
  use iso_fortran_env, only : STDERR => error_unit

  implicit none; private

  interface
    integer(kind=c_int) function setenv_c(env_var, env_val, replace) bind(c, name='setenv')
      import c_int, c_char
      character(kind=c_char), intent(in) :: env_var(*) !< Name of the variable to set
      character(kind=c_char), intent(in) :: env_val(*) !< Value to set the variable to
      integer(kind=c_int),    intent(in) :: replace    !< If 1, overwrite the value,
    end function setenv_c
  end interface

  interface
    integer(kind=c_int) function unsetenv_c( env_var ) bind(c, name='unsetenv')
      import c_int, c_char
      character(kind=c_char), intent(in) :: env_var(*) !< Name of the variable to clear
    end function unsetenv_c
  end interface


  public :: irand
  public :: use_cluster
  public :: c_str
  public :: setenv
  public :: unsetenv

  contains

  !> Returns a random integer between 0 and 255
  integer function irand()
    real :: real_rand

    call random_number(real_rand)

    irand = nint(real_rand*255)
  end function irand

  logical function use_cluster()
    character(len=16) :: server_type

    call get_environment_variable('SR_DB_TYPE', server_type)
    server_type = to_lower(server_type)
    use_cluster = .false.
    if (len_trim(server_type)>0) then
      select case (server_type)
        case ('clustered')
          use_cluster = .true.
        case ('standalone')
          use_cluster = .false.
        case default
          use_cluster = .false.
      end select
    endif
  end function use_cluster

  !> Returns a lower case version of the string. Only supports a-z
  function to_lower(str) result(lower_str)
    character(len=*),          intent(in   ) :: str !< String
    character(len = len(str)) :: lower_str

    character(26), parameter :: caps = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    character(26), parameter :: lows = 'abcdefghijklmnopqrstuvwxyz'

    integer :: i, i_low

    lower_str = str
    do i=1,len_trim(str)
      i_low = index(caps,str(i:i))
      if (i_low > 0) lower_str(i:i) = lows(i_low:i_low)
    enddo
  end function to_lower

  !> Convert a Fortran string to a C-string (i.e. append a null character)
  function c_str(f_str)
    character(len=*) :: f_str !< The original Fortran-style string
    character(kind=c_char,len=len_trim(f_str)+1) :: c_str !< The resultant C-style string

    c_str = trim(f_str)//C_NULL_CHAR
  end function c_str

  !> Set an environment variable to a given value
  subroutine setenv(env_var, env_val, replace)
    character(len=*) :: env_var !< Environment variable to set
    character(len=*) :: env_val !< The value to set the variable
    logical, optional :: replace !< If true (default) overwrite the current value

    integer(kind=c_int) :: c_replace, err_code
    character(kind=c_char, len=len_trim(env_var)+1) :: c_env_var
    character(kind=c_char, len=len_trim(env_val)+1) :: c_env_val

    c_replace = 1
    if (present(replace)) then
       if (replace)       c_replace = 1
       if (.not. replace) c_replace = 0
    endif
    c_env_var = c_str(env_var)
    c_env_val = c_str(env_val)

    err_code = setenv_c(c_env_var, c_env_val, c_replace)
    if (err_code /= 0) then
      write(STDERR,*) "Error setting", c_env_var, c_env_val
      error stop
    endif
  end subroutine setenv

  !> Clear an environment variable
  subroutine unsetenv(env_var)
    character(len=*) :: env_var !< Environment variable to clear

    integer(kind=c_int) :: err_code
    character(kind=c_char, len=len_trim(env_var)+1) :: c_env_var
    c_env_var = c_str(env_var)

    err_code = unsetenv_c(c_env_var)
    if (err_code /= 0) then
      write(STDERR,*) "Error clearing", c_env_var
      error stop
    endif
  end subroutine unsetenv

end module test_utils
