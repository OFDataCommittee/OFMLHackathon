interface
  subroutine set_model_from_file_c( c_client, key, key_length, model_file, model_file_length,      &
      backend, backend_length, device, device_length, batch_size, min_batch_size, tag, tag_length, &
      inputs, input_lengths, n_inputs, outputs, output_lengths, n_outputs ) bind(c, name="set_model_from_file")
    use iso_c_binding, only : c_ptr, c_size_t, c_int, c_char
    type(c_ptr),            intent(in) :: key               !< The key to use to place the model
    integer(kind=c_size_t), intent(in) :: key_length        !< The length of the key c-string, excluding null
    character(kind=c_char), intent(in) :: model_file(*)     !< The file storing the model
    integer(kind=c_size_t), intent(in) :: model_file_length !< The length of the file name c-string, excluding null
                                                            !! terminating character
    character(kind=c_char), intent(in) :: backend           !< The name of the backend (TF, TFLITE, TORCH, ONNX)
    integer(kind=c_size_t), intent(in) :: backend_length    !< The length of the backend name c-string, excluding null
                                                            !! terminating character
    character(kind=c_char), intent(in) :: device(*)         !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
    integer(kind=c_size_t), intent(in) :: device_length     !< The length of the device name c-string, excluding null
                                                            !! terminating character
    integer(kind=c_int),    intent(in) :: batch_size        !< The batch size for model execution
    integer(kind=c_int),    intent(in) :: min_batch_size    !< The minimum batch size for model execution
    character(kind=c_char), intent(in) :: tag(*)            !< A tag to attach to the model for information purposes
    integer(kind=c_size_t), intent(in) :: tag_length        !< The length of the tag c-string, excluding null
                                                            !! terminating character
    type(c_ptr),            intent(in) :: inputs            !< One or more names of model input nodes (TF models)
    integer(kind=c_int),    intent(in) :: input_lengths     !< The length of each input name c-string, excluding null
                                                            !! terminating character
    integer(kind=c_int),    intent(in) :: n_inputs          !< The number of inputs
    type(c_ptr),            intent(in) :: outputs           !< One or more names of model output nodes (TF models)
    integer(kind=c_int),    intent(in) :: output_lengths    !< The length of each output name c-string, excluding null
                                                            !! terminating character
    integer(kind=c_int),    intent(in) :: n_outputs         !< The number of outputs
  end subroutine set_model_from_file_c
end interface