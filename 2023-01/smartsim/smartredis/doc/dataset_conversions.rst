*******************
DataSet Conversions
*******************

Dataset conversion refers to a multi-step workflow where a user can create a SmartRedis dataset 
which can then be retrieved on the Python side, and then transformed into a {Dataformat} object.
An Xarray Dataset conversion, in this case, creates a SmartRedis dataset that is prepared to be converted 
into a an Xarray Dataarray object, and then performs the conversion into the Xarray Dataarray object.

Dataset conversion to Xarray within the DatasetConversion class is two-step process: 
#. where additional metadata is added to an existing Dataset to add the possibility of conversion and 
#. a transformation method which ingests a Dataset and returns an object in the native dataformat. 

Xarray DataSet Conversions 
==========================

The Xarray Dataset format conversion methods follow the format of the two step process, where the 
``add_metadata_for_xarray()`` method performs the task of adding additional metadata to an existing Dataset,
allowing the ``transform_to_xarray()`` method to identify which fields should be used in the construction of 
the specific data format. 

.. code-block:: python

    // add meta_data_for_xarray interface
    add_metadata_for_xarray(dataset, data_names, dim_names, coord_names=None, attr_names=None)

.. code-block:: python

    // transform_to_xarray interface
    transform_to_xarray(dataset)

Separating the adding of the metadata and the transformation into the appropriate data format minimizes 
the SmartRedis interference with the existing dataset. 

.. note::

    The ``add_metadata_for_xarray()``and ``transform_to_xarray()`` methods support adding multiple tensors into 
    SmartRedis datasets and storing their common metadata. If multiple data items are present with common metadata 
    then multiple xarrays will be built. Support for multiple data items with differing metadata is not yet supported. 


add_metadata_for_xarray
-----------------------

The ``add_metadata_for_xarray()`` method supports attaching data and metadata to a tensor within a SmartRedis dataset, 
preparing the SmartRedis Dataset for transformation. The ``add_metadata_for_xarray()`` method should not interfere with the 
existing Dataset API to extract and manipulate data.


We expect users to construct the datasets themselves using the Dataset API before calling the ``add_metadata_for_xarray()`` method.
Only the field names will be being passed into ``add_metadata_for_xarray()``, so the actual structure of the dataset and any of the metadata will 
not be affected after calling the method. 

Below is an example of the creation of a SmartRedis Dataset and addition of tensor data and metadata done by the user:

.. code-block:: python

    ds1 = Dataset("ds-1d")
    dataset.add_tensor("1ddata",data1d)
    dataset.add_tensor("x",longitude_1d)
    dataset.add_meta_string("x_coord_units",'degrees E') 
    dataset.add_meta_string("x_coord_longname",'Longitude')
    dataset.add_meta_string("units",'m/s')
    dataset.add_meta_string("longname",'velocity')
    dataset.add_meta_string("convention",'CF1.5')
    dataset.add_meta_string("dim_data_x","x")

Below is an example of the ``add_metadata_for_xarray()`` method calls to pass in field names of data and 
metadata of the created SmartRedis Dataset under the appropriate parameter names for the creation of 
the tensor data variable for the Xarray object and the coordinate data variable for the Xarray object:

.. code-block:: python

    # Calling method add_metadata_for_xarray on the created dataset
    DatasetConverter.add_metadata_for_xarray(
        ds1, 
        data_names=["1ddata"],
        dim_names=["dim_data_x"],
        coord_names=["x"],
        attr_names=["units","longname","convention"]
    )
    # Calling method add_metadata_for_xarray for longitude coordinate
    DatasetConverter.add_metadata_for_xarray(
        ds1,
        data_names=["x"], 
        dim_names=["dim_data_x"],
        attr_names=["x_coord_units","x_coord_longname"] 
    )

The ``add_metadata_for_xarray()`` method has the ability to define the coordinates of each dimension of a tensor added to the dataset 
(e.g. the actual x, y, z values of every element of a 3D tensor or vector of timestamps for a 1D timeseries) 
If the user would like to add data variables as coordinates to their Xarray DataArray, the data name of the data variable
must match the name of the coordinate_name being specified in the ``add_metadata_for_xarray()`` parameters, and the method will recognize the appropriately named data variable
as a coordianate variable to be added to the Xarray DataArray. 

The ability to extract data (metadata,tensors, etc.) by their original field names remains intact after any call to 
``add_metadata_for_xarray()``.

The ``add_metadata_for_xarray()`` method uses metadata names that are reserved by and on behalf of the ``add_metadata_for_xarray()`` method:

.. code-block:: python

    "_xarray_data_name"
    "_xarray_dim_name"
    "_xarray_coord_name" 
    "_xarray_attr_name" 

.. note:: 

    Calling the ``add_metadata_for_xarray()`` method to add the reserved metadata names is necessary for the ``transform_to_xarray()`` method 
    to read the metadata names and unpack the data for the data format conversion. 

transform_to_xarray
-------------------

The ``transform_to_xarray()`` converts from a SmartRedis dataset into a dictionary of keys as the name of the Xarray DataArray, and the values
as the actual converted Xarray DataArrays.  

The transform to xarray method will retrieve the field names store in the Dataset under these metadata names 
for populating the native xarray conversion to DataArray method. 

.. code-block:: python

    xarray_ret = DatasetConverter.transform_to_xarray(ds1)

An example of the returned dictionary of the ``transform_to_xarray()`` method: 

.. code-block:: python

    {'1ddata': <xarray.DataArray '1ddata' (x: 10)>
    array([0.75239102, 0.87698733, 0.57916855, 0.59621001, 0.22552972,
        0.17998833, 0.27133364, 0.3092101 , 0.82813876, 0.00731646])
    Coordinates:
    * x        (x) float64 0.0 40.0 80.0 120.0 160.0 200.0 240.0 280.0 320.0 360.0
    Attributes:
        units:       m/s
        longname:    velocity
        convention:  CF1.5}

