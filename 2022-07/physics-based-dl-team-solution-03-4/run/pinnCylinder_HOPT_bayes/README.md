# To run the case
Initialise the environment where libtorch and openfoam with alias `of2112` have been installed:
`source ../../setup_torch.sh`
`of2112`

Following above steps, run the following command to just generate the data for default parameters:
`cp -r 0_OF_orig 0 && blockMesh && pinnPotentialFOAM` 

# To run the hyper parameter optimisation
create a virtual environment using venv/virtualenv
Then install the dependencies for Bayesian optimisation using the following command:
`pip install -r requirements.txt`

Now, initialise the environment and test the installation using IPython:

```
import BayesianOptimization as bopt
print(bopt.__version__)
```


Then run the hyperparameter optimisation script `bayes_opt_pinn.py  <N_iter>' either from the command line or from the ipython shell. 
