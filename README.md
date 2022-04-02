# pybasic

pybasic is a python package for background and shading correction of microscopy images. It implements the algorithm as described in the below reference.

> Peng, T., Thorn, K., Schroeder, T. et al. A BaSiC tool for background and shading correction of optical microscopy images. Nat Commun 8, 14836 (2017). https://doi.org/10.1038/ncomms14836

A MATLAB implementation from the above lab is [available here](https://github.com/marrlab/BaSiC).

## Installation

```
pip install git+https://github.com/biomicrodev/pybasic.git
```

We recommend installing in a virtual environment.

## Usage

To generate the flatfield (and optionally, darkfield) correction images:

```
chmod u+x bin/pybasic-gen
bin/pybasic-gen ...
```

To correct images using a given flatfield (and darkfield) image:

```
chmod u+x bin/pybasic-correct
bin/pybasic-correct ...
```

## Examples

Download the following examples from the MATLAB repository's [Dropbox link](https://www.dropbox.com/s/plznvzdjglrse3h/Demoexamples.zip?dl=0).

TODO: add jupyter notebooks for the examples.
