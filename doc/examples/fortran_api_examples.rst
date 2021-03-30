.. _fortran_client_examples:

*******
Fortran
*******


In this section, examples are presented using the SILC Fortran
API to interact with the RedisAI tensor, model, and script
data types.  Additionally, an example of utilizing the
SILC ``DataSet`` API is also provided.

.. note::

    The Fortran API examples rely on the ``SSDB`` environment
    variable being set to the address and port of the Redis database.

.. note::

    The Fortran API examples are written
    to connect to a non-cluster Redis database.  Update the
    ``Client`` constructor call to connect to a Redis cluster.

Tensors
=======

The SILC Fortran client is used to communicate between
a Fortran client and the Redis database. In this example,
the client will be used to send an array to the database
and then unpack the data into another Fortran array.

This example will go step-by-step through the program and
then present the entirety of the example code at the end.

**Importing and declaring the SILC client**

The SILC client must be declared as the derived type
``client_type`` imported from the ``silc_client`` module.

.. code-block:: fortran

  program example
    use silc_client, only : client_type

    type(client_type) :: client
  end program example

**Initializing the SILC client**

The SILC client needs to be initialized before it can be used
to interact with the database. Within Fortran this is
done by calling the type-bound procedure
``initialize`` with the input argument ``.true.``
if using a clustered database or ``.false.`` otherwise.

.. code-block:: fortran

  program example
    use silc_client, only : client_type

    type(client_type) :: client

    call client%initialize(.false.) ! Change .true. to false if using a clustered database
  end program example

**Putting a Fortran array into the database**

After the SILC client has been initialized,
a Fortran array of any dimension and shape
and with a type of either 8, 16, 32, 64 bit
``integer`` or 32 or 64-bit ``real`` can be
put into the database using the type-bound
procedure ``put_tensor``.
In this example, as a proxy for model-generated
data, the array ``send_array_real_64`` will be
filled with random numbers and stored in the
database using ``put_tensor``. This subroutine
requires the user to specify a string used as the
'key' (here: ``send_array``) identifying the tensor
in the database, the array to be stored, and the
shape of the array.

.. literalinclude:: ../../examples/serial/fortran/silc_put_get_3D.F90
  :linenos:
  :language: fortran
  :lines: 1-11,13-24,26-27

**Unpacking an array stored in the database**

'Unpacking' an array in SILC refers to filling
a Fortran array with the values of a tensor
stored in the database.  The dimensions and type of
data of the incoming array and the pre-declared
array are checked within the client to
ensure that they match. Unpacking requires
declaring an array and using the ``unpack_tensor``
procedure.  This example generates an array
of random numbers, puts that into the database,
and retrieves the values from the database
into a different array.

.. literalinclude:: ../../examples/serial/fortran/silc_put_get_3D.F90
  :linenos:
  :language: fortran


Datasets
========

The following code snippet shows how to use the Fortran
Client to store and retrieve dataset tensors and
dataset metadata scalars.

.. literalinclude:: ../../examples/serial/fortran/silc_dataset.F90
  :linenos:
  :language: fortran