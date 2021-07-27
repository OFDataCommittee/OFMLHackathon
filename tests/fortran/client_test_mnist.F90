program mnist_test

  use smartredis_client, only : client_type
  use test_utils,  only : use_cluster

  implicit none

  character(len=*), parameter :: model_key = "mnist_model"
  character(len=*), parameter :: model_file = "../../cpp/mnist_data/mnist_cnn.pt"
  character(len=*), parameter :: script_key = "mnist_script"
  character(len=*), parameter :: script_file = "../../cpp/mnist_data/data_processing_script.txt"

  type(client_type) :: client
  integer :: err_code
  character(len=2) :: key_suffix

  call client%initialize_client(use_cluster())

  call client%set_model_from_file(model_key, model_file, "TORCH", "CPU")
  call client%set_script_from_file(script_key, "CPU", script_file)

  call run_mnist(client, model_key, script_key)

contains

subroutine run_mnist( client, model_name, script_name )
  type(client_type), intent(in) :: client
  character(len=*),  intent(in) :: model_name
  character(len=*),  intent(in) :: script_name

  integer, parameter :: mnist_dim1 = 28
  integer, parameter :: mnist_dim2 = 28
  integer, parameter :: result_dim1 = 10

  real, dimension(1,1,mnist_dim1,mnist_dim2) :: array
  real, dimension(1,result_dim1) :: result

  character(len=255) :: in_key
  character(len=255) :: script_out_key
  character(len=255) :: out_key

  character(len=255), dimension(1) :: inputs
  character(len=255), dimension(1) :: outputs

  ! Construct the keys used for the specifiying inputs and outputs
  in_key = "mnist_input"
  script_out_key = "mnist_processed_input"
  out_key = "mnist_processed_input"

  ! Generate some fake data for inference
  call random_number(array)
  call client%put_tensor(in_key, array, shape(array))

  ! Prepare the script inputs and outputs
  inputs(1) = in_key
  outputs(1) = script_out_key
  call client%run_script(script_name, "pre_process", inputs, outputs)
  inputs(1) = script_out_key
  outputs(1) = out_key
  call client%run_model(model_name, inputs, outputs)
  result(:,:) = 0.
  call client%unpack_tensor(out_key, result, shape(result))

  print *, "Result: ", result
  print *, "Fortran test mnist: passed"

end subroutine run_mnist

end program mnist_test
