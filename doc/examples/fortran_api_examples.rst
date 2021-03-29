.. _fortran_client_examples:

***********
Fortran API
***********

The SILC Fortran API is written using an object-oriented approach with two derived types ``client_type`` and
``dataset_type``. The examples shown here demonstrate the basic functionality and use of these SILC types by
constructing simple Fortran programs. Code examples will be built up gradually, with the full program source code
shown at the end of each section.

Importing SILC
==============
The public facing parts of SILC-Fortran are contained in two modules ``silc_client`` and ``silc_dataset``. These can be imported into Fortran code in the usual way:

.. code-block:: fortran

  program example
    use silc_dataset, only : dataset_type
    use silc_client,  only : client_type
  end program example

.. note::

  ``dataset_type`` and ``client_type`` are the only public elements of these modules

Example: Sending and unpacking an array using the Fortran client
================================================================

The SILC Fortran client is used to communicate between a Fortran client and the Redis database. In this example, the
client will be used to send an array to the database and then unpack the data into another Fortran array. The client requires that the environment variable ``SSDB`` be set prior to the execution of the SILC-enabled program. This variable specifies the address and port of the primary Redis database node. Using the default settings on a local cluster,

.. code-block:: bash

  export SSDB="127.0.0.1:6379;"

Importing and declaring the SILC client
---------------------------------------

The SILC client must be declared as the derived type ``client_type`` imported from the ``silc_client`` module.

.. code-block:: fortran

  program example
    use silc_client, only : client_type

    type(client_type) :: client
  end program example

Initializing the SILC client
----------------------------

The SILC client needs to be initialized before it can be used to interact with the database. Within Fortran this is
done by calling the type-bound procedure ``initialize`` with the input argument ``.true.`` if using a clustered
database or ``.false.`` otherwise.

.. code-block:: fortran

  program example
    use silc_client, only : client_type

    type(client_type) :: client

    call client%initialize(.false.) ! Change .true. to false if using a clustered database
  end program example

Putting a Fortran array into the database
-----------------------------------------

After the SILC client has been initialized, a Fortran array of any dimension and shape and with a type of either 8, 16, 32, 64 bit
``integer`` or 32 or 64-bit ``real`` can be put into the database using the type-bound procedure
``put_tensor``.
In this example, as a proxy for model-generated data, the array ``send_array_real_64`` will be filled with
random numbers and stored in the database using ``put_tensor``. This subroutine requires the user
to specify a string used as the 'key' (here: ``send_array``) identifying the tensor in the database,
the array to be stored, and the shape of the array.

.. literalinclude:: ../../examples/serial/fortran/silc_put_get_3D.F90
  :linenos:
  :language: fortran
  :lines: 1-11,13-24,26-27

Unpacking an array stored in the database
-----------------------------------------

'Unpacking' an array in SILC refers to filling a Fortran array with the values of a tensor stored in the database.
The dimensions and type of data of the incoming array and the pre-declared array are checked within the client to
ensure that they match. Unpacking requires declaring an array and using the ``unpack_tensor`` procedure.
This example generates an array of random numbers, puts that into the database, and
retrieves the values from the database into a different array.

.. literalinclude:: ../../examples/serial/fortran/silc_put_get_3D.F90
  :linenos:
  :language: fortran


Datasets
========

The following code snippet shows how to use the Fortran Client to store and retrieve dataset tensors and
dataset metadata scalars.

.. literalinclude:: ../../examples/serial/fortran/silc_dataset.F90
  :linenos:
  :language: fortran