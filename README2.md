# OpenFOAM MachineLearning Hackathon- Team 3 & 4
 
The objective was to implement a data driven and physics driven neural network for inferring potential flow given sparse data points from the bulk. 
This was achieved by integrating inbuilt functions of the PyTorch C++ API with OpenFOAM.

### Team 3 Members
   - Ryley McConkey [Email](rmcconke@uwaterloo.ca "rmcconke@uwaterloo.ca")
   - Junsu Shin [Email](junsu.shin@unibw.de "junsu.shin@unibw.de")
   - Reza Lotfi [Email](rezalotfi127@gmail.com "rezalotfi127@gmail.com")

### Team 4 Members
   - Rahul Sundar [Email](rahulsundar95@smail.iitm.ac.in "rahulsundar95@smail.iitm.ac.in")
   - Abhijeet Vishwasrao [Email](abhijeet.vishwasrao@polytechnique.edu "abhijeet.vishwasrao@polytechnique.edu")
   - Biniyam Sishah [Email](biniyamsishah@gmail.com "biniyamsishah@gmail.com")

### Supervisors
   - Tomislav Maric
   - Andre Weiner   

## Dependencies
For the OpenFOAM and libtorch installation check README.md

For the other python scripts used here in the repository:
`python >= 3.7`
`requirements.txt` - for Bayesian Optimisation

### Procedure
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
physics-based-dl/
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
---README.md
---Allrun
---Allmake
---Allclean
---Presentation.pdf
---ProjectReport.pdf
```

### Code Changes in OpenFOAM
According to the physics of the problem and solver, there has been changes made in different dictionaries.




### Optimization
Bayesian Optimization has been implemented to the solver.  
### How to use
Follow this procedure to run the code and see the results.

open the terminal and navigate _Cylinder_ testcase and execute the command below:

```bash
cd run/unit_box_domain/cylinder

./Allclean

./Allrun

pinnPotentialFoam

paraFoam

```

## Results
**CPU time** and **accuracy** of using the NN as the MLP without the PiNN residual and the PiNN approach for approximating Psi ,u is being compared.
