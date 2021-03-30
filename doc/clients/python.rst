******
Python
******

Client API
==========

.. note::

  The Python client documentation is incomplete.

.. currentmodule::  smartredis

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

.. autoclass:: Client
   :members:
   :show-inheritance:


DataSet API
===========

.. currentmodule::  smartredis

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

