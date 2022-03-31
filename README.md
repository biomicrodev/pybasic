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
python bin/pybasic-gen ...
```

To correct images using a given flatfield (and darkfield) image:

```
python bin/pybasic-correct ...
```

## Examples

Download the following examples from the MATLAB repository's [Dropbox link](https://www.dropbox.com/s/plznvzdjglrse3h/Demoexamples.zip?dl=0).

TODO: add jupyter notebooks for the examples.

## Notes

To avoid reading in all images, stacking, and then resizing, we use dask to resize on-the-fly before stacking. We assume that the dtypes and shapes of all images match that of the first image. Unfortunately, the iterative nature of the algorithm makes it difficult to truly parallelize, but the resizing step makes a reasonable tradeoff, as the flatfield and darkfield images are smooth (sparse in k-space).

Correction: a distributed form of LADMAP is possible, and therefore for this algorithm, but that seems like a big rabbit hole.

Currently, there is no limit to the size of the image stack; this may change in the future.

### The algorithm
So, the algorithm *should* work. I've tried my best to understand the supplementary materials in the original publication and the LADMAP paper that this method uses. 

I have also renamed variables as reasonably as I could, without being too long, and without using single letters.

Some modifications that I don't yet understand are:
- the initial penalty value
- the penalty multiplier
- ent1 and ent2; these may be related to convergence rates?

It's possible these were empirically determined by Peng et al.

I would not expect exact agreement, especially because of some floating point errors for the stop condition; it may be stopping prematurely? I am not sure, but the results are *very* close while being several times faster. I will compute the difference for the provided example images soon.

There is also a computation potentially missing from the publication's implementation: in `inexact_alm_rspca_l1`, $I^{B,k}$ is not being recomputed when updating the Lagrange multiplier. It doesn't seem to be that big of a deal though; correcting it had only a slight effect.
