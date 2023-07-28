# Multi Objective Optimization on OpenFOAM cases

![](https://zenodo.org/badge/611991004.svg)
> If you're using this piece of software, please care enough to [cite it](https://zenodo.org/record/7997394) in your publications

Relying on [ax-platform](https://ax.dev) to experiment around 0-code parameter variation and multi-objective optimization
of OpenFOAM cases.

## Objectives and features
- Parameter values are fetched through a YAML/JSON configuration file. Absolutely no code should be needed, add parameters
  to the YAML file and they should be picked up automatically
- The no-code thing is taken to the extreme, through a YAML config file, you can (need-to):
  - Specify the template case
  - Specify how the case is ran
  - Specify how/where parameters are substituted
  - Specify how your metrics are computed

## The scripts

- `paramVariation.py` runs a Parameter Variation Study with the provided config and outputs trial data as a CSV file.
  Trial parameters are generated with SOBOL.
- `multiObjOpt.py` runs a multi-objective optimization study (can use the same config file) and captures a JSON snapshot
  of the experiment. You can also run it with a single objective.

## How do I use this?

The only requirement of usage (aside from being able to install dependency software) is that your template case needs to
be **ready to be parameterized**.

```bash
# Clone the repository
git clone https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization multiOptFoam
cd multiOptFoam
# Install dependencies
pip3 install -r requirements.txt
# Run parameter variation with config.yaml on pitzDaily case
./paramVariation.py
# Or run a multi-objective optimization
./multiObjOpt.py
```

## Docs and Sample configuration

You can find a quick tutorial in the [docs page](docs.md). The [sample config file](config.yaml) and [case](pitzDaily)
are also documented.

## Contribution is welcome!

By either filing issues or opening pull requests, you can contribute to the development
of this project, which I would appreciate.
