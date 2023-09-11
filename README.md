# Understanding Metastasis Organotropism Patterns Through Within-cell and Between-cells Molecular Interaction Networks

* AUTHOR: [João A. I. Miranda¹](https://github.com/joanismi/)
* SUPERVISOR: [Francisco R. Pinto¹](https://github.com/frpinto)
* CONTACT: jamiranda@ciencias.ulisboa.pt

¹ [RNA Systems Biology Lab](https://github.com/GamaPintoLab) 


Code to reproduce methods &amp; results from my Master's Thesis Project.
I will add a link to the final document as soon as it is officially published.

To reproduce this project just clone or download this repository and create a Python virtual environment.
To learn how to create a virtual environment refer to [creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

If you use our data or analysis in your research, please cite us!

## System Requirements
Our code was run using the following software:
- Python version 3.10.12
- R version 4.1.2

#### Python packages to install:
See requirements.txt

To install all packages using this file just type
```
pip install requirements.txt
```
inside your python virtual environment

#### R packages to install:
To install the IRkernel to run R in notebooks follow these [instructions](https://irkernel.github.io/installation/).
Required packages can be installed using notebook 7.
---

## Analysis notebooks:
The notebooks in the `src` directory can be used to generate the data required to create all figures in present in the thesis. Just run the notebooks in the following order:
- 1_gene_expression_analysis.ipynb: this Python notebook covers Methods sections 2.2 - 2.5 and subsection 2.7.2 and Results section 3.1.
- 2_organotropism_pairs.ipynb: this Python notebook covers Methods section 2.6 and Results section 3.2.
- 3_intercell_data_analysis.ipynb: this Python notebook covers Methods subsection 2.7.1.
- 4_intercell_networks.ipynb: this Python notebook covers Methods subsections 2.7.3 - 2.7.5 and Results section 3.3.
- 5_intercell_interactions_analysis.ipynb: this Python notebook covers Methods section 2.8 and Results section 3.4.
- 6_intracell_network.ipynb: this Python notebook covers Methods 2.9 and Results section 3.5.
- 7_go_enrichment_analysis.ipynb: this R notebook covers Results subsection 3.5.2.
- 8_cdg_enrichment_analysis.ipynb: this Python notebook covers Results subsection 3.5.1.
