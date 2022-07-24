# OpenFOAM Workshop 17 Training - Towards physics-based deep learning in OpenFOAM: Combining OpenFOAM with the PyTorch C++ API 

## Author
 
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

### Usage 

Navigate to the *unit_box_domain* test case and execute the command below:

```
cd run/unit_box_domain
cp -r 0.org 0 && blockMesh && pinnFoamSetSphere && pinnFoam 
```

### Visualization 

In *unit_box_domain*, open the ParaView state file:

```
paraview --state=visualize.pvsm 
```

To view training loss diagrams `jupyter notebook` and open `pinnFoam-visualize-single-training.ipynb', then execute `Re-run and Clear Data`. 

### Grid Search

A primitive Grid Search using `python.subprocess` is implemented in `run/pinnFoam-grid-search.ipynb`, just re-run this Jupyter notebook. Visualization of the Grid Search is done by `run/pinnFoam-visualize.ipynb`. 

## License

GPL v3.0
