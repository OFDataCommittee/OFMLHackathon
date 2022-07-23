# OpenFOAM Workshop 17 Training - Towards physics-based deep learning in OpenFOAM: Combining OpenFOAM with the PyTorch C++ API 

## Author
 
Tomislav Maric, MMA, Mathematics Department, TU Darmstadt, maric@mma.tu-darmstadt.de

## Installation 

### Dependencies 

* OpenFOAM-v2112, compiled with the [C++20 standard](https://gcc.gnu.org/projects/cxx-status.html#cxx20)
    * To compile OpenFOAM with `-std=c++2a`, edit `$WM_PROJECT_DIR/wmake/rules/General/Gcc/c++` and use `CC = g++-10 -std=c++2a` to compile OpenFOAM. 
* python Pandas, matplotlib

### Installation 

```
   ofw17-training-physics-based-dl> ./Allwmake
```

### Usage 

In `ofw17-training-physics-based-dl/run/unit_box_domain`

```
    unit_box_domain > blockMesh && pinnFoamSetSphere && pinnFoam 
```

### Visualization 

Run 

```
    unit_box_domain > paraview --state=visualize.pvsm 
```

To view training loss diagrams `jupyter notebook` and open `pinnFoam-visualize-single-training.ipynb', then execute `Re-run and Clear Data`. 

### Grid Search

A primitive Grid Search using `python.subprocess` is implemented in `run/pinnFoam-grid-search.ipynb`, just re-run this Jupyter notebook. Visualization of the Grid Search is done by `run/pinnFoam-visualize.ipynb`. 

## License

GPL v3.0
