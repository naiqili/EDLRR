# EDLRR
  Official implementation of "Efficient Differentiable Approximation of Generalized Low-rank Regularization" (IJCAI 2025) 
  [[arxiv](https://arxiv.org/abs/2505.15407)]
  
  Low-rank regularization (LRR) has been widely applied in various machine learning tasks, but the associated optimization is challenging. Directly optimizing the rank function under constraints is NP-hard in general. To overcome this difficulty, various relaxations of the rank function were studied. However, optimization of these relaxed LRRs typically depends on singular value decomposition, which is a time-consuming and nondifferentiable operator that cannot be optimized with gradient-based techniques. To address these challenges, in this paper we propose an efficient differentiable approximation of the generalized LRR. The considered LRR form subsumes many popular choices like the nuclear norm, the Schatten-$p$ norm, and various nonconvex relaxations. Our method enables LRR terms to be appended to loss functions in a plug-and-play fashion, and the GPU-friendly operations enable efficient and convenient implementation. Furthermore, convergence analysis is presented, which rigorously shows that both the bias and the variance of our rank estimator rapidly reduce with increased sample size and iteration steps. In the experimental study, the proposed method is applied to various tasks, which demonstrates its versatility and efficiency. 

## Code Structure
  - ``./Demo-image.ipynb``: Demo code for the image restoration experiment.
  - ``./Demo-bgDetection.ipynb``: Demo code for the fore-background separation experiment.
  - ``./series_expansion/``: Scripts for calculating series expansions with Mathematica.
