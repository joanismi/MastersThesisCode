# Understanding Metastasis Organotropism Patterns Through Within-cell and Between-cells Molecular Interaction Networks

* AUTHOR: João A. I. Miranda¹ 
* SUPERVISOR: Francisco R. Pinto¹
* CONTACT: [](jamiranda@ciencias.ulisboa.pt)

¹ RNA Systems Biology Lab


Code to reproduce methods &amp; results from my Master's Thesis Project.


If you use our data or analysis in your research, please cite our research article!


## System Requirements
Our code was run using the following software:
- Python version 3.10.12
- R version 4.1.2

#### Python packages to install:


#### R packages to install:


## Analysis notebooks:
The notebooks in the `src` directory can be used to generate all data  
The notebook 0_reproduce_results_from_raw_data.ipynb is a good starting place if one wants to explore our data themselves. More detailed versions of our analysis code is contained in the following notebooks:
- Ig_genes.ipynb details how we determined which genes fall in the immunoglobulin loci, to remove them from downstream analyses.
- 4a_puritywork-published.ipynb contains our analysis of sample purity (% tumor cells in sample) using our Bayesian purity model. It also contains the code to generate Fig. 2a from our paper. 
- 4d_limma.ipynb contains our limma-voom differential expression analysis comparing malignant or pre-malignant pseudobulk samples vs. normal pseudobulk samples.
- 5_NMF_rawdata-moreHVG-published.ipynb contains code related to generating the input data for SignatureAnalyzer and our analysis of SignatureAnalyzer results, including Figs. 3a-d from our paper.
- 5b_heterogeneity.ipynb contains code for analyzing the heterogeneity of signature expression within tumor samples, including Fig. 4c from our paper.
- helper_functions_published.py contains functions that are used throughout the other notebooks included in the repo.
