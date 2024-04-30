## IceCube_kNN_examples ##

The repository contains the Python codes, data, and Jupyter notebooks that illustrate the computations <br>
of some key quantities and reproduce some of the figures in the paper:

Title: *High-energy Neutrino Source Cross-correlations with Nearest Neighbor Distributions*<br>
Authors: Zhuoyang Zhou, Jessi Cisewski-Kehe, Ke Fang, and Arka Banerjee.

Please download the entire repository into the same folder before running the notebooks. 

Note that almost all experiments in the original paper are done with a high throughput computing cluster. <br>
Therefore, we only provide simpler versions of the codes without any parallelization and some computed data (like the cross-correlations 
at a specific f_astro) obtained from our experiments.

We thank Michael Larson for his help with using the IceCube public ten-year point-source data.<br>
Part of the codes, including utils.py, background_generator.py, and signal_generator.py, used in the paper 
for astrophysical and atmospheric events and repositories are from the GitHub repository created by Michael Larson: https://github.com/mjlarson/I3PublicDataSampler<br>

Please visit the repository for a more detailed illustration of events generation.

### Purposes and Descriptions for each Notebook:
1. <ins>*IceCube_kNN_CSRxCSR*</ins>: This is to illustrate how to compute the kNN-CDFs, joint kNN-CDFs, and kNN-CDFs cross-correlation on a group of complete spatial randomness
data (CSR data) on the surface of a unit sphere. It will compute and plot the cross-correlations between uncorrelated samples (two different CSR data) and
correlated data (two subsamples from the same group of CSR data).

2. <ins>*IceCube_kNN_Events_Generation*</ins>: This notebook illustrate the synthetic events generation (include maskings) for atmospheric-only events and a combined atm. + astro. sample
with f_astro ~ 0.0374. It contains the plotting for the synthetic astrophysical neutrinos and background-only events with aitoff projection.

3. <ins>*IceCube_kNN_LRT-MLE*</ins>: This notebook illustrate how to conduct the likelihood ratio test with the likelihood function defined in the paper and the maximum likelihood
estimation procedure. We provide 1000 cross-correlations between the selected WISE-2MASS sample wth synthetic data with f_astro ~ 0.0374 for illustration. 

4. <ins>*IceCube_kNN_Real_Measurement*</ins>: This notebook reproduces the Figure 6 in the paper by computing the cross-correlation between the IceCube 10-year data: 2008 - 2018 and the selected WISE-2MASS sample. 
