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

### Generate

To generate the flatfield (and optionally, darkfield) correction images:

```
pybasic-gen [-h] 
            [--and-darkfield] 
            [--iter-dims N [N ...]] 
            [--rgb]
            [--flatfield-reg VALUE]
            [--darkfield-reg VALUE]
            [--working-size VALUE]
            images [images ...]
```

To support a variety of image dimensionalities, the `--iter-dims` argument allows specifying which dimensions are not image dimensions; basic operates over all iter dims. For instance, to correct a set of 10 x 5 x 512 x 512 images (in TCYX format), `--iter-dims` is `0 1`, and the output is a flatfield/darkfield image of the same shape (10 x 5 x 512 x 512).

This script will take in a set of images and output a flatfield image (and optionally a darkfield image). The output images are float64, which may be large. In the future, if there is reason to, an option could be added to keep the output images at the working size.

### Correct

To correct images using a given flatfield (and darkfield) image:

```
chmod u+x bin/pybasic-correct
bin/pybasic-correct ...
```

## Examples

| Image set     | Link                                     |
|:--------------|:-----------------------------------------|
| Cell culture  | [notebook](examples/cell_culture.ipynb)  |

To run the jupyter notebook examples, download sample images from the MATLAB repository's [Dropbox link](https://www.dropbox.com/s/plznvzdjglrse3h/Demoexamples.zip?dl=0), and place into `examples/images`.

## Tests

Run `pytest`.

## Todo

* clean up CLI outputs according to [current best practices](https://clig.dev/)
* set up logging
* modernize python packaging code
* add timelapse correction code