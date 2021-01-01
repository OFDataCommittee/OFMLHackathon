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

.. code-block:: python
  :linenos:

  from silc import Client
  import numpy as np

  # initialize the client (keep connections alive)
  client = Client(False, False)

  # Send a 2D Tensor
  key = "2D_array"
  array = np.random.randint(-10, 10, size=(10, 10))
  client.put_tensor(key, array)

  # Get the 2D Tensor
  returned_array = client.get_tensor("2D_array")

Client Initialization
---------------------

The Python client connection is initialized with the object constructor.
The optional boolean argument ``cluster`` indicates whether the client
will be connecting to a single database node or multiple distributed
nodes which is referred to as a cluster.


Python Client API
=================


.. currentmodule::  silc

.. autosummary::

    Client.__init__
    Client.put_tensor
    Client.get_tensor

.. autoclass:: Client
   :members:
   :show-inheritance:
   :inherited-members:


Python Dataset API
==================

.. currentmodule::  silc

.. autosummary::

    Dataset.__init__
    Dataset.add_tensor
    Dataset.get_tensor

.. autoclass:: Dataset
   :members:
   :show-inheritance:
   :inherited-members:

