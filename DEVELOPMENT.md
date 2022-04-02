# Notes

To avoid reading in all images, stacking, and then resizing, we use dask to resize on-the-fly before stacking. We assume that the dtypes and shapes of all images match that of the first image. Unfortunately, the iterative nature of the algorithm makes it difficult to truly parallelize, but the resizing step makes a reasonable tradeoff, as the flatfield and darkfield images are smooth (sparse in $k$-space).

Correction: a distributed form of LADMAP is possible, and therefore for this algorithm, but that seems like a big rabbit hole.

Currently, there is no limit to the size of the image stack; this may change in the future.

## The algorithm
So, the algorithm *should* work. I've tried my best to understand the supplementary materials in the original publication and the [LADMAP paper](https://arxiv.org/abs/1109.0367) that this method uses.

I have also renamed variables as reasonably as I could, without being too long, and without using single letters.

Some modifications that I don't yet understand are:
- the initial penalty value
- the penalty multiplier
- ent1 and ent2; these may be related to convergence rates?

It's likely these were empirically determined by the authors.

I would not expect exact agreement, especially because of some floating point errors for the stop condition; it may be stopping prematurely? I am not sure, but the results are *very* close while being several times faster. I will compute the difference for the provided example images soon.

There is also a line potentially missing from the publication's implementation: in `inexact_alm_rspca_l1`, $I^{B,k}$ is not being recomputed when updating the Lagrange multiplier, even though the pseudocode in the supplementary section mentions it. It doesn't seem to be that big of a deal though; correcting it had only a slight effect.
