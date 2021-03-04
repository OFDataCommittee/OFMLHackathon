module example_utils

  implicit none; private

  public :: irand
  public :: use_cluster

  contains

  !> Returns a random integer between 0 and 255
  integer function irand()
    real :: real_rand

    call random_number(real_rand)

    irand = nint(real_rand*255)
  end function irand

  logical function use_cluster()

    character(len=16) :: silc_test_cluster

    call get_environment_variable('SILC_TEST_CLUSTER', silc_test_cluster)
    silc_test_cluster = to_lower(silc_test_cluster)
    if (len_trim(silc_test_cluster)>0) then
      select case (silc_test_cluster)
        case ('true')
          use_cluster = .true.
        case ('false')
          use_cluster = .false.
        case default
          use_cluster = .false.
      end select
    endif

  end function use_cluster

  !> Returns a lower case version of the string. Only supports a-z
  function to_lower( str ) result(lower_str)
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

end module example_utils
