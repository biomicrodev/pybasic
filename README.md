# pybasic

pybasic is a python package for background and shading correction of microscopy images. It implements the algorithm as described in the below reference.

> Peng, T., Thorn, K., Schroeder, T. et al. A BaSiC tool for background and shading correction of optical microscopy images. Nat Commun 8, 14836 (2017). https://doi.org/10.1038/ncomms14836

A MATLAB implementation from the above lab is [available here](https://github.com/marrlab/BaSiC).

## Installation

```
pip install git+https://github.com/biomicrodev/pybasic.git
```

We recommend installing in a virtual environment.

## Examples

| Image set    | Link                                    |
|:-------------|:----------------------------------------|
| Cell culture | [notebook](examples/cell_culture.ipynb) |
| WSI brain    | [notebook](examples/wsi_brain.ipynb)    |

To run the jupyter notebook examples, download sample images from the MATLAB repository's [Dropbox link](https://www.dropbox.com/s/plznvzdjglrse3h/Demoexamples.zip?dl=0), and place into `examples/images`.
