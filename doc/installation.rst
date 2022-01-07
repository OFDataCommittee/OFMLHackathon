************
Installation
************

SmartRedis clients are intended to be used as a library linked into other
applications.  For C, C++, and Fortran, the clients
can be compiled as a library that is linked with an application
at compile time. For Python, the clients can be used just like
any other pip library.

Before installation, it is recommended to use an OS and compiler that are known to be reliable with SmartRedis.

SmartRedis is tested with the following operating systems on a daily basis:

.. list-table::
    :widths: 50
    :header-rows: 1
    :align: center

    * - OS (tested daily)
    * - MacOS
    * - Ubuntu


SmartRedis is tested with the following compilers on a daily basis:

.. list-table::
    :widths: 50
    :header-rows: 1
    :align: center

    * - Compilers (tested daily)
    * - GNU (GCC/GFortran)
    * - Intel (icc/icpc/ifort)
    * - Apple Clang


SmartRedis has been tested with the following compiler in the past, but on a less regular basis as the compilers listed above:

.. list-table::
    :widths: 50
    :header-rows: 1
    :align: center

    * - Compilers (irregularly tested in the past)
    * - Cray Clang


SmartRedis has been used with the following compilers in the past, but they have not been tested. We do not imply that these compilers work for certain:

.. list-table::
    :widths: 50
    :header-rows: 1
    :align: center

    * - Compilers (used in the past, but not tested)
    * - Cray Classic
    * - NVHPC
    * - PGI


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
