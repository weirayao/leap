# LEAP: Learning Temporally Causal Latent Processes from General Temporal Data

### Overview
In this project, we consider both a nonparametric, nonstationary setting and a parametric setting for the latent processes and propose two provable conditions under which temporally causal latent processes can be identified from their nonlinear mixtures. We propose LEAP, a theoretically-grounded architecture that extends Variational Autoencoders (VAEs) by enforcing our conditions through proper constraints in causal process prior. Experimental result on various data sets demonstrate that temporally causal latent processes are reliably identified from observed variables under different dependency structures and that our approach considerably outperforms baselines that do not leverage history or nonstationarity information. This is one of the first works that successfully recover time-delayed latent processes from nonlinear mixtures without using sparsity or minimality assumptions. 

**Our Approach:** we leverage **nonstationarity** in process noise or **functional and distributional forms** of temporal statistics to identify temporally causal latent processes from observation.
<p align="center">
  <img align="middle" src="https://github.com/weirayao/leap/blob/main/imgs/motivation.jpg" alt="relational inference" width="800"/>
</p>

<!-- *In addition to structure, our approach allows inferring Granger-causal effect signs*:
<p align="center">
  <img align="middle" src="https://github.com/i6092467/GVAR/blob/master/images/scheme_panel_2.png" alt="interpretable relational inference" width="5000"/>
</p>
 -->
**Framework**: LEAP consists of (A) encoders and decoders for specific data types; (B) a recurrent inference network that approximates the posteriors of latent causal variables, and (C) a causal process prior network that models nonstationary latent causal processes with independent noise (IN) condition constraints.
<p align="center">
  <img align="middle" src="https://github.com/weirayao/leap/blob/main/imgs/overall.jpg" width="700"/>
</p>

### Experiments
Experiment results are showcased in Jupyter Notebooks in `/tests` folder. Each notebook contains the scripts for analysis and visualization for one specific experiment.

<p align="center">
  <img align="middle" src="https://github.com/weirayao/leap/blob/main/imgs/np_syn.png" width="600"/>
</p>

Run the scripts in `/leap/scripts` to generate results for experiment.

Further details are documented within the code.

### Requirements
To install it, create a conda environment with `Python>=3.7` and follow the instructions below. Note, that the current implementation of LEAP requires a GPU.
```
conda create -n leap python=3.7
cd leap
pip install -e .
```

### Datasets

- Synthetic data: `python leap/datasets/gen_dataset.py `
- KittiMask: https://github.com/bethgelab/slow_disentanglement
- Mass-Spring system: https://yunzhuli.github.io/V-CDN/
- CMU Mocap databse: http://mocap.cs.cmu.edu/
