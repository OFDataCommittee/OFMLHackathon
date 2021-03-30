**************************************************
Online inference in C++ with a PyTorch CNN and MPI
**************************************************


This example uses the SILC C++ client to provide an example of a PyTorch convolutional neural net and
pre-processing script loaded into the Redis database.  The pre-processing script and
PyTorch model are executed with sample data.

Source Code
===========

C++ program
-----------

.. literalinclude:: ../../examples/parallel/cpp/silc_mnist.cpp
  :linenos:
  :language: C++

Python Pre-Processing
---------------------

.. literalinclude:: ../../examples/common/mnist_data/data_processing_script.txt
  :linenos:
  :language: Python
  :lines: 15-20