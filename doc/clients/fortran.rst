
*******
Fortran
*******

Overview
========

The SILC Fortran interface is centered around two Fortran modules: ``silc_client`` and ``silc_dataset``. The only
public element of these modules are, respectively, ``client_type`` and ``dataset_type``. These derived types take
advantage of Fortran object-oriented features by having procedure-bound methods that implement most of the
SILC functionality. (see :ref:`Unsupported SILC Features <unsupported_silc_features>`). Other than these derived types, all inputs
and outputs from functions and subroutines are Fortran primitives (e.g. ``real``, ``integer``, ``character``).

32-bit and 64-bit ``real`` and 8, 16, 32, and 64-bit signed ``integer`` arrays (tensors) are supported. All
procedures are overloaded to avoid needing to specify the type-specific subroutine.

To communicate with the Redis client, SILC-Fortran relies on Fortran/C/C++ interoperability to wrap the methods of
the C++ client. All transformations from Fortran constructs to C constructs are handled within the SILC client itself
(e.g enforcing Fortran/C types and column-major to row-major arrays). No conversions need to be done within the
application.

The example below shows the code required to send and receive data with the Fortran client.

.. literalinclude:: ../../examples/serial/fortran/silc_put_get_3D.F90
  :language: fortran
  :linenos:

Compiler Requirements
---------------------

Fortran compilers need to support the following features

* Object-oriented programming support (Fortran 2003)
* Fortran-C interoperability, ``iso_c_binding`` (Fortran 2003)
* Assumed rank (``dimension(..)``) arrays (Fortran 2018)

These language features are supported by Intel 19, GNU 9, and Cray 8.6 and later versions.

.. _unsupported_silc_features:

Unsupported SILC features
-------------------------
Due to limitations of C/Fortran interoperability, some of the features in the Python, C, and C++ clients have not
been implemented. This includes

* Retrieving metadata strings (Dataset: ``get_meta_strings``)
* Returning a dataset tensor or tensor from the database as an opaque type (Dataset: ``get_dataset_tensor``, Client: ``get_tensor``)
* Getting tensors from the database as an opaque type (Client:``get_tensor``) (note unpacking tensors into allocated
  memory is supported, see the :ref:`Fortran Client Examples section<fortran_client_examples>`)

Source code organization
========================
SILC-Fortran source code is contained within the following files

* ``client.F90``: Contains the ``client_type`` and all associated methods
* ``dataset.F90`` Contains the ``dataset_type`` and all associated methods
* ``fortran_c_interop.F90``: Routines to aid in Fortran/C interoperability

The ``client.F90`` and ``dataset.F90`` files are further broken down into a number of 'included' files to prevent
duplicated code and organize the variety of methods included within each type. The naming conventions are prefixed by
general functionality and suffixed by the type of code contained within.

* ``<functionality>_interfaces.inc``: Define the C-bound interfaces to the SILC-C methods
* ``<functionality>_methods_common.inc``: Represents the source code that is exactly the same for all methods which

share the same functionality, but differ only by the type of variable

For example, ``client/put_tensor_interfaces.inc`` define the Fortran-C interfaces to put a tensor into the database.
``client/put_tensor_methods_common.inc`` form the main body of the source code that handles the conversion and
calling of the Fortran-C interfaces.


API Reference
=============

.. note ::
   Fortran autodoc-ing in Sphinx is relatively primitive, however the code has been doxygenized and is built along with the rest of the documentation. They can be found in ``silc/doc/fortran_client/html``.

Fortran Client API
------------------

The following are overloaded interfaces which support 32/64-bit ``real`` and 8, 16, 32, and 64-bit ``integer`` tensors

* ``put_tensor``
* ``unpack_tensor``

.. f:automodule:: silc_client

Fortran Dataset API
-------------------

The following are overloaded interfaces which support 32/64-bit ``real`` and 8, 16, 32, and 64-bit ``integer`` tensors

* ``add_tensor``
* ``unpack_dataset_tensor``

Similarly the following interfaces are overloaded to support 32/64-bit ``real`` and ``integer`` metadata

* ``add_meta_scalar``
* ``get_meta_scalar``

.. f:automodule:: silc_dataset

.. _fortran_client_examples:

Fortran Client Examples
=======================

This section contains examples showing how the Fortran client can be used to interact with the database.

Datasets
--------

The following code snippet shows how to use the Fortran Client to store and retrieve dataset tensors and
dataset metadata scalars.

.. literalinclude:: ../../examples/serial/fortran/silc_dataset.F90
  :linenos:
  :language: fortran


Models
------

The following example shows how to store, retrieve, and use a pre-processing script and a DL model in the database with the Fortran Client. 
The model and the script are stored as files in the ``../../../common/mnist_data/`` path relative to the compiled executable.

.. literalinclude:: ../../examples/serial/fortran/silc_mnist.F90
  :linenos:
  :language: fortran