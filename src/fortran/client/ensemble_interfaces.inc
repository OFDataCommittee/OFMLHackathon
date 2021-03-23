interface
  subroutine set_data_source_c( client, source_id, source_id_length ) bind(c, name="set_data_source")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t
    type(c_ptr),            value :: client
    character(kind=c_char)        :: source_id(*)
    integer(kind=c_size_t), value :: source_id_length
  end subroutine set_data_source_c
end interface

interface
  subroutine use_model_ensemble_prefix_c( client, use_prefix) bind(c, name="use_model_ensemble_prefix")
    use iso_c_binding, only : c_ptr, c_bool
    type(c_ptr),            value :: client
    logical(kind=c_bool),   value :: use_prefix
  end subroutine use_model_ensemble_prefix_c
end interface

interface
  subroutine use_tensor_ensemble_prefix_c( client, use_prefix) bind(c, name="use_tensor_ensemble_prefix")
    use iso_c_binding, only : c_ptr, c_bool
    type(c_ptr),            value :: client
    logical(kind=c_bool),   value :: use_prefix
  end subroutine use_tensor_ensemble_prefix_c
end interface