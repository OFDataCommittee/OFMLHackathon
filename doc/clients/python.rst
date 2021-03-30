******
Python
******

Client API
==========

.. currentmodule::  silc

.. autosummary::

    Client.__init__
    Client.put_tensor
    Client.get_tensor
    Client.put_dataset
    Client.get_dataset
    Client.set_function
    Client.set_script
    Client.set_script_from_file
    Client.get_script
    Client.run_script
    Client.set_model
    Client.set_model_from_file
    Client.get_model
    Client.run_model
    Client.model_exists
    Client.key_exists
    Client.poll_key
    Client.poll_tensor
    Client.poll_model
    Client.set_data_source
    Client.use_model_ensemble_prefix
    Client.use_tensor_ensemble_prefix

.. autoclass:: Client
   :members:
   :show-inheritance:


DataSet API
===========

.. currentmodule::  silc

.. autosummary::

    Dataset.__init__
    Dataset.add_tensor
    Dataset.get_tensor
    Dataset.add_meta_scalar
    Dataset.get_meta_scalars
    Dataset.add_meta_string
    Dataset.get_meta_strings

.. autoclass:: Dataset
   :members:
   :show-inheritance:

