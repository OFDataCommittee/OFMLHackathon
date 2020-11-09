====
SILC
====

The SmartSim Infrastructure Library Clients (SILC) are a library
of redis-based clients built primarily for use with SmartSim.

.. list-table::
   :widths: 15 10 30
   :header-rows: 1

   * - Language
     - Version/Standard
     - Status
   * - Python
     - 3.7+
     - In development
   * - C++
     - C++17
     - Stable
   * - C
     - C99
     - In development
   * - Fortran
     - Fortran 2003 +
     - Awaiting development

Project Status
==============

Currently the C++ is the only mature implementation of the SILC clients. The
C and Python client are in development and should be done soon.


Using SILC
==========

Installation
------------

To build the dependencies for the C++ client, invoke the ``build_deps.sh``
script from within the root directory of this repository

.. code-block:: bash

    source build_deps.sh

Examples
--------

For examples of how to build and use the c++ client, refer to the c++ test
directory in this repository.

