
***
C
***

Using the C Client
==================

The SmartSim C client allows users to send and receive data from
other SmartSim entities stored in the database. The code snippet below shows
the code required to send and receive data with the C client. In the
following subsections, general groups of functions that are provided by the
Python client API will be described.

.. literalinclude:: ../../examples/serial/c/example_put_get_3D.c
  :language: C
  :linenos:

In the above example, ``result`` points to a memory area managed by the client. 
In most cases, pre-allocating memory and unpacking a tensor in it can be
more suitable. Please refer to the :ref:`Tensor section of the Data Strucutres
documentation page <data_structures_tensor>` for more details. In the 
following example, a 1D tensor is allocated and put in the database, then it 
is retrieved and unpacked in a pre-allocated area of memory in the user code.

.. literalinclude:: ../../examples/serial/c/example_put_unpack_1D.c
  :language: C
  :linenos:

C Client API
============

.. doxygenfile:: c_client.h
   :project: c_client


C Dataset API
=============

.. doxygenfile:: c_dataset.h
   :project: c_client

