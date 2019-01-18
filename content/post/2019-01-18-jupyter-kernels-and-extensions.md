---
title: Jupyter Kernels and Extensions
author: Bjoern Koneswarkanatha
date: '2019-01-18'
slug: jupyter-kernels-and-extensions
categories:
  - jupyter
tags:
  - conda
  - jupyter
---


# No Code


# nb_conda_kernels

`nb_conda_kernels` will allow you to select different kernels for your jupyter notebooks from the GUI

1. Install `nb_conda_kernels` in the environment that you start `jupyter notebook` from

```shell
conda install nb_conda_kernels
```

2. Install `ipykernel` in the environments that you want to be able to select
```
source activate <env>
conda install ipykernel
```

# papermill