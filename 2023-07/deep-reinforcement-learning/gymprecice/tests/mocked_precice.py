from unittest.mock import MagicMock


action_read_iteration_checkpoint = MagicMock()
action_write_iteration_checkpoint = MagicMock()
action_write_initial_data = MagicMock()


class Interface:
    """
    Mock class to represent python-bindings 'precice' in all tests.
    """

    def __init__(self, name, config_file, rank, procs):
        pass

    def initialize(self):
        raise NotImplementedError

    def advance(self, *args):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def get_dimensions(self):
        raise NotImplementedError

    def get_mesh_id(self):
        raise NotImplementedError

    def get_data_id(self, *args):
        raise NotImplementedError

    def initialize_data(self, *args):
        raise NotImplementedError

    def set_mesh_vertices(self, *args):
        raise NotImplementedError

    def is_action_required(self):
        raise NotImplementedError

    def imark_action_fulfilled(self):
        raise NotImplementedError

    def is_coupling_ongoing(self):
        raise NotImplementedError

    def is_time_window_complete(self):
        raise NotImplementedError

    def requires_initial_data(self):
        raise NotImplementedError

    def requires_reading_checkpoint(self):
        raise NotImplementedError

    def requires_writing_checkpoint(self):
        raise NotImplementedError

    def read_block_vector_data(self, *args):
        raise NotImplementedError

    def read_vector_data(self, *args):
        raise NotImplementedError

    def read_block_scalar_data(self, *args):
        raise NotImplementedError

    def read_scalar_data(self, *args):
        raise NotImplementedError

    def write_block_vector_data(self, *args):
        raise NotImplementedError

    def write_vector_data(self, *args):
        raise NotImplementedError

    def write_block_scalar_data(self, *args):
        raise NotImplementedError

    def write_scalar_data(self, *args):
        raise NotImplementedError
