************
Installation
************

SmartRedis clients are intended to be used as a library linked into other
applications.  For C, C++, and Fortran, the clients
can be compiled as a library that is linked with an application
at compile time. For Python, the clients can be used just like
any other pip library.

SmartRedis is ran on these compilers and OS'es regularly:


.. list-table::
    :widths: 50 50 50
    :header-rows: 1
    :align: center

    * - OS
      - Compiler
      - Compiler Versions
    * - MacOS
      - Clang
      - 12
    * - Ubuntu 20.04
      - GCC/GFortran
      - 8 - 10
    * - Ubuntu 20.04
      - Intel
      - 2021.4.0


This document will show how to:
  1. Install the SmartRedis Python client from the release
  2. Build and install SmartRedis as a static lib from release
  3. Build SmartRedis from source

Build and Install SmartRedis Python Client from Release
=======================================================

.. include:: ./install/python_client.rst

----------------------------------------------

Build SmartRedis Library from Release
=====================================

.. include:: ./install/lib.rst

-----------------------------------------------

Build SmartRedis from Source
============================

.. include:: ./install/from_source.rst
