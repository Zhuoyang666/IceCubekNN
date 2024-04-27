#IceCube_kNN_examples
The repository contains the Python codes, data, and Jupyter notebooks that illustrate the computations of some key 
quantities and reproduce some of the figures in the paper:

Title: *High-energy Neutrino Source Cross-correlations with Nearest Neighbor Distributions*
Authors: Zhuoyang Zhou, Jessi Cisewski-Kehe, Ke Fang, Arka Banerjee.

Please download the entire repository into the same folder before running the notebooks. 





Note that almost all experiments in the original paper are done with a high throughput computing cluster. Therefore, 
we only provide simpler versions of the codes without any parallelization and some computed data (like the cross-correlations 
at a specific f_astro) obtained from our experiments.

We thank Michael Larson for his help with using the IceCube public ten-year point-source data.
Part of the codes, including utils.py, background_generator.py, and signal_generator.py, used in the paper and repository are from the 
Github repository created by Michael Larson: https://github.com/mjlarson/I3PublicDataSampler
