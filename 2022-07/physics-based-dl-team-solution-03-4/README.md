# OpenFOAM Workshop 17 Training - Towards physics-based deep learning in OpenFOAM: Combining OpenFOAM with the PyTorch C++ API 
 
The objective was to implement a data driven and physics driven neural network for inferring potential flow given sparse data points from the bulk. 
This was achieved by integrating inbuilt functions of the PyTorch C++ API with OpenFOAM.


## Authors:

### Team 3 Members
   - Ryley McConkey [Email](rmcconke@uwaterloo.ca "rmcconke@uwaterloo.ca")
   - Junsu Shin [Email](junsu.shin@unibw.de "junsu.shin@unibw.de")
   - Reza Lotfi [Email](rezalotfi127@gmail.com "rezalotfi127@gmail.com")

### Team 4 Members
   - Rahul Sundar [Email](rahulsundar95@smail.iitm.ac.in "rahulsundar95@smail.iitm.ac.in")
   - Abhijeet Vishwasrao [Email](abhijeet.vishwasrao@polytechnique.edu "abhijeet.vishwasrao@polytechnique.edu")
   - Biniyam Sishah [Email](biniyamsishah@gmail.com "biniyamsishah@gmail.com")

## Mentors
Tomislav Maric, MMA, Mathematics Department, TU Darmstadt, maric@mma.tu-darmstadt.de


## Installation 

### Dependencies 

* OpenFOAM-v2206, compiled with the [C++20 standard](https://gcc.gnu.org/projects/cxx-status.html#cxx20)
    * to compile OpenFOAM with `-std=c++2a`, edit `$WM_PROJECT_DIR/wmake/rules/General/Gcc/c++` and use `CC = g++-10 -std=c++2a` to compile OpenFOAM.
    * the pre-built binary work as well on most systems, e.g., if you follow these [installation instructions](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian)
    * older version starting *v1912* should work as well
* python Pandas, matplotlib

### Installation 

At the top-level of this repository, run:
```
source setup-torch.sh
./Allwmake
```
This script will:
- check if required environment variables are available
- download *LibTorch* if needed (PyTorch C++ API)
- compile everything in *applications*


For the OpenFOAM and libtorch installation check README.md

For the other python scripts used here in the repository:
`python >= 3.7`
`requirements.txt` - for Bayesian Optimisation

### Procedure to implement pinnPotentialFoam
1. Copy the `pinnFoam` application into the `pinnPotentialFoam` application, rename it + compile it.
   - Rename the `pinnFoam` to `pinnPotentialFoam`.
   - Rename all `pinnFoams` in the code to `pinnPotentialFoam`. `grep -r pinnFoam`
2. Edit `createFields.H` and read the potientialFoam fields **Phi** and **U**.
3. Adapt the Neural Network (NN) $\Psi(x,y,z,\theta)$ to map a point in space $\boldsymbol{x} =(x,y,z)$ to the output vector $O=(\Phi,U_x, U_y, U_z)$, with $\boldsymbol{U}=(U_x, U_y, U_z)$ being the potential-flow velocity, and $\Phi$ the velocity-potential. 
4. Remove the existing PiNN residual MSE and train the NN as a Multilayer Perceptron on the Phi and U fields computed by OpenFOAM’s `potentialFoam` solver.
5. Extend the NN into a PiNN for potential flow, by programming the potential-flow PDE residual 
    
    This means: 
   - Implementing the Laplace operator for $\Phi$.
   - Implementing the divergence operator for $\boldsymbol{U}$.
   - Combining both into the residual MSE, and summing the residual MSE with the data MSE.


### File structure

```
physics-based-dl-solution-03-4/
---applications/
   ---pinnFoam
   ---pinnFoamSetSphere
   ---dnnPotentialFoam
   ---pinnPotentialFoam
---run/
   ---Cylinder
   ---dnnCylinder
   ---pinnCylinder
   ---dnnCylinderHOPT_grid
   ---dnnCylinderHOPT_bayes
   ---pinnCylinderHOPT_grid
   ---pinnCylinderHOPT_bayes
---schematics/
   ---OFMLHackathon.draw.io 
   ---<Other schematic plots>.png 
---plots/
   ---<loss convergence plots>.png
   ---dnn/<Flow field plots>.png
   ---pinn/<Flow field plots>.png
---README.md
---Allrun
---Allmake
---Allclean
---Presentation.pdf
---ProjectReport.pdf (PENDING)
```
The `draw.io` file contains the modifiable source for all the schematics gerated for this project. 

### Usage 

Navigate first to the *Cylinder* to run the *dnnCylinder* test case and execute the command below:

```
cd run/Cylinder
./Allrun
cp -r 0 ../dnnCylinder/0
cd ../dnnCylinder
./Allrun
```

### Visualization 

In *dnnCylinder*, open paraView:

```
touch dnnCylinder.foam
paraview dnnCylinder.foam 
```
Please remember to uncheck skip zero time step option before refreshing and selecting `apply` option in paraview. 

To view training loss diagrams `jupyter notebook` and open `pinnFoam-visualize-single-training.ipynb', then execute `Re-run and Clear Data`. 

### Grid Search

A primitive Grid Search using `python.subprocess` is implemented in `run/pinnCylinder_HOPT_grid/pinnPotentialFoam-grid-search.ipynb`, just re-run this Jupyter notebook. Visualization of the Grid Search is done by `run/pinnCylinder_HOPT_grid/pinnPotentialFoam-visualize.ipynb`. 


A primitive Grid Search using `python.subprocess` is implemented in `run/dnnCylinder_HOPT_grid/dnnPotentialFoam-grid-search.ipynb`, just re-run this Jupyter notebook. Visualization of the Grid Search is done by `run/dnnCylinder_HOPT_grid/dnnPotentialFoam-visualize.ipynb`. 

A bayesian optimization based  hyperparameter search using `python.subprocess` is implemented in `run/pinnCylinder_HOPT_bayes/bayes_opt_pinn.py` and `run/dnnCylinder_HOPT_bayes/bayes_opt_dnn.py`.

## License

GPL v3.0
