program main

use smartredis_client, only : client_type
use test_utils,  only : setenv, use_cluster
use iso_fortran_env, only : STDERR => error_unit

implicit none

character(len=255) :: ensemble_keyout, ensemble_keyin
character(len=255) :: tensor_key, model_key, script_key
character(len=255) :: script_file, model_file

real, dimension(10) :: tensor
type(client_type) :: client

ensemble_keyout = "producer_0"
call setenv("SSKEYIN", "producer_0,producer_1")
call setenv("SSKEYOUT", ensemble_keyout)

call client%initialize_client(use_cluster())
call client%use_model_ensemble_prefix(.true.)

! Put tensor, script, and model into the database. Then check to make
! sure that keys were prefixed correctly

tensor_key = "ensemble_tensor"
call client%put_tensor(tensor_key, tensor, shape(tensor))
if (.not.client%tensor_exists(tensor_key)) then
  write(STDERR,*) 'Tensor does not exist: ', tensor_key
  error stop
endif
if (.not.client%key_exists(trim(ensemble_keyout)//"."//trim(tensor_key))) then
  write(STDERR,*) 'Key does not exist: ', trim(ensemble_keyout)//"."//tensor_key
  error stop
endif

script_key = "ensemble_script"
script_file = "../../cpp/mnist_data/data_processing_script.txt"
call client%set_script_from_file(script_key, "CPU", script_file)
if (.not.client%model_exists(script_key)) then
  write(STDERR,*) 'Script does not exist: ', script_key
  error stop
endif

model_key = "ensemble_model"
model_file = "../../cpp/mnist_data/mnist_cnn.pt"
call client%set_model_from_file(model_key, model_file, "TORCH", "CPU")
if (.not.client%model_exists(model_key)) then
  write(STDERR,*) 'Model does not exist: ', model_key
  error stop
endif

call client%destructor()

! Now set the consumer tests
ensemble_keyout = "producer_1"
call setenv("SSKEYIN", "producer_1,producer_0")
call setenv("SSKEYOUT", ensemble_keyout)

call client%initialize_client(use_cluster())
call client%use_model_ensemble_prefix(.true.)

! Check that the keys associated with producer_1 do not exist
if (client%tensor_exists(tensor_key)) then
  write(STDERR,*) "Tensor should not exist: ", tensor_key
  error stop
endif
if (client%key_exists(trim(ensemble_keyout)//"."//trim(tensor_key))) then
  write(STDERR,*) "Key should not exist: "//trim(ensemble_keyout)//"."//tensor_key
  error stop
endif
call client%set_data_source("producer_0")
if (.not. client%tensor_exists(tensor_key)) then
  write(STDERR,*) "Tensor does not exist: ", tensor_key
  error stop
endif
if (.not. client%model_exists(model_key)) then
  write(STDERR,*) "Model does not exist: "//model_key
  error stop
endif
if (.not. client%model_exists(script_key)) then
  write(STDERR,*) "Script does not exist: "//script_key
  error stop
endif


end program main
