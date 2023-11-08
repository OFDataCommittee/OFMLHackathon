# Compile the application 

1. [Install smartredis + smartsim](https://www.craylabs.org/docs/installation_instructions/basic.html#smartredis) 
1. Install OpenFOAM and source OpenFOAM environment. 
2. Activate conda environment.  

```
smartsim> source configure-smartredis.sh
smartsim> cd streaming-dmd
streaming-dmd> wmake foamSmartSimDmd
```

# Running the application with SmartSim

From within `cavity` case

```
streaming-dmd> jupyter notebook foam-smartsim-dmd.ipynb
```

Click on Run->Run all cells.
