***
C++
***

Using the C Client
==================

The SmartSim C++ client allows users to send and receive data from
other SmartSim entities stored in the database. The code snippet below shows
the code required to send and receive data with the C++ client. In the
following subsections, general groups of functions that are provided by the
C++ client API will be described.

.. literalinclude:: ../../examples/serial/cpp/silc_put_get_contiguous_3D.cpp
  :language: C++
  :lines: 68-73,82-95
  :lineno-start: 68
  :linenos:

In the above example, ``g_nested_result`` and ``g_contig_result`` point to memory areas
managed by the client: ``g_nested_result`` is a nested tensor, whereas
``g_contig_result`` represents the same tensor, but stored in a contiguous area of memory.

In most cases, pre-allocating memory and unpacking a tensor in it can be
more suitable. Please refer to the :ref:`Tensor section of the Data Strucutres
documentation page <data_structures_tensor>` for more details. In the 
following example, a 3D tensor is allocated and put in the database, then it 
is retrieved and unpacked in a pre-allocated area of memory in the user code.

.. literalinclude:: ../../examples/serial/cpp/silc_put_get_contiguous_3D.cpp
  :language: C++
  :lines: 68-73,74-79
  :lineno-start: 68
  :linenos:

Similar to the first example, here ``u_nested_result`` is a nested tensor, whereas
``u_contig_result`` represents the same tensor, but stored in a contiguous area of memory.

The complete source code for the examples can be found in the 
`C++ example directory of the repository`_. Further examples are available in the 
:ref:`C++ API examples section <cpp_client_examples>`.

.. _C++ example directory of the repository: https://github.com/CrayLabs/SILC/examples/serial/cpp/

C++ client API
==============

.. doxygenclass:: SILC::Client
   :project: cpp_client
   :members:
   :undoc-members:


C++ Dataset API
===============

.. doxygenclass:: SILC::DataSet
   :project: cpp_client
   :members:
   :undoc-members:

