******
Python
******

Using the Python Client
=======================

The SmartSim Python client allows users to send and receive data from
other SmartSim entities stored in the database. The code snippet below shows
the code required to send and receive data with the Python client. In the
following subsections, general groups of functions that are provided by the
Python client API will be described.

.. literalinclude:: ../../examples/serial/python/example_put_get_tensor.py
  :language: python
  :linenos:

Client Initialization
---------------------

The Python client connection is initialized with the object constructor.
The optional boolean argument ``cluster`` indicates whether the client
will be connecting to a single database node or multiple distributed
nodes which is referred to as a cluster.

An address can be provided to the initialization of the client as well.
This address should be a string with an ip address and port separated
by a colon. If an address is not provided, the client will search
for the ``SSDB`` environment variable.

Further examples can be found in the 
:ref:`Python API examples <python_client_examples>` section.

Python Client API
=================

.. note::

  The Python client documentation is incomplete.

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

.. autoclass:: Client
   :members:
   :show-inheritance:


Python Dataset API
==================

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

