************
Fortran APIs
************

The following page provides a comprehensive overview of the SmartRedis Fortran
Client and Dataset APIs.
Further explanation and details of each are presented below.

Client API
==========

The Client API is purpose-built for interaction with the backend database,
which extends the capabilities of the Redis in-memory data store.
It's important to note that the SmartRedis Client API is the exclusive
means for altering, transmitting, and receiving data within the backend
database. More specifically, the Client API is responsible for both
creating and modifying data structures, which encompass :ref:`Models <data-structures-model>`,
:ref:`Scripts <data-structures-script>`, and :ref:`Tensors <data-structures-tensor>`.
It also handles the transmission and reception of
the aforementioned data structures in addition to :ref:`Dataset <data-structures-dataset>`
data structure. Creating and modifying the ``DataSet`` object
is confined to local operation by the DataSet API.

The following are overloaded interfaces which support
32/64-bit ``real`` and 8, 16, 32, and 64-bit ``integer`` tensors

* ``put_tensor``
* ``unpack_tensor``

.. f:automodule:: smartredis_client

Dataset API
===========

The Fortran DataSet API enables a user to manage a group of tensors
and associated metadata within a datastructure called a ``DataSet`` object.
The DataSet API operates independently of the database and solely
maintains the dataset object in-memory. The actual interaction with the Redis database,
where a snapshot of the DataSet object is sent, is handled by the Client API. For more
information on the ``DataSet`` object, click :ref:`here <data-structures-dataset>`.

The following are overloaded interfaces which support
32/64-bit ``real`` and 8, 16, 32, and 64-bit
``integer`` tensors

* ``add_tensor``
* ``unpack_dataset_tensor``

Similarly the following interfaces are overloaded to
support 32/64-bit ``real`` and ``integer`` metadata

* ``add_meta_scalar``
* ``get_meta_scalar``

.. f:automodule:: smartredis_dataset

API Notes
=========

Fortran autodoc-ing in Sphinx is relatively primitive, however
the code has been doxygenized and is built along with the rest
of the documentation. They can be found in
``smartredis/doc/fortran_client/html``.

Importing SmartRedis
--------------------
The public facing parts of SmartRedis-Fortran are contained in two
modules ``smartredis_client`` and ``smartredis_dataset``. These can
be imported into Fortran code in the usual way:

.. code-block:: fortran

  program example
    use smartredis_dataset, only : dataset_type
    use smartredis_client,  only : client_type
  end program example

.. note::

  ``dataset_type`` and ``client_type`` are the
  only public elements of these modules

Using the Fortran Client
------------------------

The SmartRedis Fortran interface is centered around two Fortran
modules: ``smartredis_client`` and ``smartredis_dataset``. The only
public element of these modules are, respectively,
``client_type`` and ``dataset_type``. These derived types take
advantage of Fortran object-oriented features by
having procedure-bound methods that implement most of the
SmartRedis functionality. (see :ref:`Unsupported SmartRedis
Features <unsupported_smartredis_features>`). Other than
these derived types, all inputs
and outputs from functions and subroutines are
Fortran primitives (e.g. ``real``, ``integer``,
``character``).

32-bit and 64-bit ``real`` and 8, 16, 32, and
64-bit signed ``integer`` arrays (tensors) are
supported. All procedures are overloaded
to avoid needing to specify the type-specific subroutine.

To communicate with the Redis client,
SmartRedis-Fortran relies on Fortran/C/C++
interoperability to wrap the methods of
the C++ client. All transformations
from Fortran constructs to C constructs
are handled within the SmartRedis client itself
(e.g enforcing Fortran/C types and column-major
to row-major arrays). No conversions need to be
done within the application.

The example below shows the code required to
send and receive data with the Fortran client.

.. literalinclude:: ../../examples/serial/fortran/smartredis_put_get_3D.F90
  :language: fortran
  :linenos:

Other examples are shown in the Fortran client examples sections.

Compiler Requirements
---------------------

Fortran compilers need to support the following features

* Object-oriented programming support (Fortran 2003)
* Fortran-C interoperability, ``iso_c_binding`` (Fortran 2003)

These language features are supported by Intel 19, GNU 9, and Cray 8.6 and later versions. Nvidia compilers
have been shown to work, but should be considered a fragile feature for now

.. _unsupported_smartredis_features:

Unsupported SmartRedis features
-------------------------------
Due to limitations of C/Fortran interoperability, some of the features in the Python, C, and C++ clients have not
been implemented. This includes

* Retrieving metadata strings (Dataset: ``get_meta_strings``)
* Returning a dataset tensor or tensor from the database as an opaque type (Dataset: ``get_dataset_tensor``, Client: ``get_tensor``)
* Getting tensors from the database as an opaque type (Client:``get_tensor``) (note unpacking tensors into allocated
  memory is supported, see the Fortran client examples section.

Source code organization
------------------------
SmartRedis-Fortran source code is contained within the following files

* ``client.F90``: Contains the ``client_type`` and all associated methods
* ``dataset.F90`` Contains the ``dataset_type`` and all associated methods
* ``fortran_c_interop.F90``: Routines to aid in Fortran/C interoperability

The ``client.F90`` and ``dataset.F90`` files are further broken down into a number of 'included' files to prevent
duplicated code and organize the variety of methods included within each type. The naming conventions are prefixed by
general functionality and suffixed by the type of code contained within.

* ``<functionality>_interfaces.inc``: Define the C-bound interfaces to the SmartRedis-C methods
* ``<functionality>_methods_common.inc``: Represents the source code that is exactly the same for all methods which share the same functionality, but differ only by the type of variable

For example, ``client/put_tensor_interfaces.inc`` define the Fortran-C interfaces to put a tensor into the database.
``client/put_tensor_methods_common.inc`` form the main body of the source code that handles the conversion and
calling of the Fortran-C interfaces.
