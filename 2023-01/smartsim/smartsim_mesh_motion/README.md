1. Source the OpenFOAM environment. 
2. Execute ./Allrun in the spinningDisk case to create the mesh and decompose the domain.
3. Make sure that the smartredis source and library folders on your machine correspond to the environmental variables set in `configure-smartredis.sh`.
4. Configure smartredis variables for compiling OpenFOAM applications and libraries,

?> source configure-smartredis.sh

5. Build the SmartSim OpenFOAM mesh motion solver

?>  wmake libso displacementSmartSimMotionSolver/

6. Start the jupyter notebook in a conda environment with installed SmartSim 0.5.0 and Python <= 3.9


?> python -m notebook 


7. Open the notebook smartsim-mesh-motion.ipynb 

