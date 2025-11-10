# Coded function object with SmartRedis support

This function object can be added to the case dynamically following these steps:
1. `source SOURCEME.sh`: This is important to set necessary environment variables. It will point to the SmartRedis installation inn
  `2023-01/smartsim/smartredis` which freezes the SmartRedis version we work with.
  - There is also a `$FOAM_CODE_TEMPLATES` environment variable which is important for the next step
  - This will also compile SmartRedis, and the OpenFOAM libraries
2. To your OpenFOAM case, add the following to define a function object that sends pressure and velocity fields to SmartRedis:
```cpp
functions
{
    smartRedis
    {
        type coded;
        libs (codedRedisFunctionObject);

        name redisAI;
        clusterMode off;
        codeWrite #{
            Info << "------ BEGIN codeWrite ------" << endl;
            sendField<volScalarField>("p");
            sendField<volVectorField>("U");
            Info << "------ END   code Write -----" << endl;
        #};
    }
}
```

All that remains is to run SmartSim. Make sure port 8000 is available and run:
```bash
python openfoam-smartsim-function-object.py
```
