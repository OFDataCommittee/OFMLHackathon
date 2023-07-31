# Case improvements

-  Better ways to parametrize the lower wall patch?
        1.  Better alternatives for OpenSCAD/cfMesh/ParaView?
               -> OpenSCAD: easy math ops and extruding from 2D polygons
               -> cfMesh: Retain cell size for mesh-dependent turbulence models...
               -> ParaView: Connectivity filter (scripts written for serial post-processing!)
        2.  Alternatives to Bezier Curves?
-  [Dependent params](depdent params) on number of curve control points? -> STAGE 02
-  More reliable pressure drop computation?
-  How to handle diverging cases?


# Code Improvements

1.  Parameter variation
    -  linspace choice parameters??
          -> Use [Uniform] instead of [SOBOL]?
          -> Or have a per-parameter strategy?
    -  list/vector types for parameters; restricted by Ax
    -  Non-OpenFOAM parameters (eg. Git commits, OpenSCAD vars!)
            1.  Workaround: Set the parameter in an OpenFOAM dict, read it and use it somewhere else
            2.  Best option: Expose parameter values in config commands maybe?
2.  Bayesian Optimization
    -  Fix and get single-objective results
    -  Verify model convergence on user-supplied parameters
    -  Penalizing vs. proper "discarded trial" status (similar to TTL-terminated trials)
    -  More control over used optimization models and transitions?
    -  Experiment restarts
    -  Better stopping strategy
3.  UDFs for metric evaluation???
4.  Online visualization?
    -  Live metric evaluation
    -  Latest n trials in pictures
    -  Seperate plotting process? Or better integration?
5.  Do smth with SmartSim/SmartRedis, get better initial conditions maybe?


# Stuff to learn more about

1.  Optimization models; mainly [Ax Models](https://ax.dev/api/modelbridge.html#ax.modelbridge.registry.Models)
    -  GPEI
    -  BO
    -  FullyBayesian(MOO) Check [This paper](https://arxiv.org/pdf/2203.12597.pdf) for SAASBO
    -  [SOBOL](SOBOL)
    -  [Uniform](Uniform) (Useful for parameter variation???)
2.  Objective threshold effects
3.  Relative feature importances of [dependent params]?
4.  Confidence in Pareto front values


# Hackathon preparation

    -  Install dependencies
```bash
     ssh -XC ubuntu@ec2-54-171-136-230.eu-west-1.compute.amazonaws.com
     # Create Conda env. (Python 3.8, requirements.txt)
     # Install OpenFOAM/cfMesh
     # Install Headless ParaView
     # Install OpenSCAD
     # --------------------------------------------------------------------------------- #
     conda activate /fsx/bias/bias_env 
     cd /fsx/bias/<your-name>
     git clone https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization omoo
     cd omoo
     # --------------------------------------------------------------------------------- #
     pip3 install -r requirements.txt
```

-  Swich to Branch `ml_hackathon_07_2023`
    -  If you want to see images of latest trials:
```bash
     # Get a free API key from imgbb.com
     export IMGBB_API_KEY=<YOUR-IMGBB-KEY>
```
    -  If you want to see the dashboard locally:
```bash
# If your dashboard is running on port 8888 from the head node
ssh -N -L localhost:8888:localhost:8888 ubuntu@ec2-54-171-136-230.eu-west-1.compute.amazonaws.com
```
