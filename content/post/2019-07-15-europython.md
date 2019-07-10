---
date : 2019-07-15
slug : europython
title : Europython 2019
author : Bjoern Koneswarakantha
categories: 
  - python
  - conference
tags: 
  - python
  - Europython
  - Pydata
  - Data Science
summary : 
thumbnailImagePosition : left
thumbnailImage : https://pbs.twimg.com/profile_images/1082571058832703488/1QVxP99r.jpg
---

# Europython 2019

## Training Days

### General
- Docker widely used to setup training environment (can take long to download)
- VS Code popular IDE

### REST API/Microservices
- use `connexion` with `swagger` to build YAML configurable REST API's. `swagger` provides documentation and user interface based on yaml.

### Pytest
- pytest can run tests from other testing suites, in order to incorporate examples from docstrings doctest can be used as `pytest --doctest`
- pytest goes by no API is a good API
- markers can be used to organise tests
- fixtures can be used to pass data to tests

- `hypothesis` can be used to generate random testing strings. If a fail is  detected it will give a minimal reproducible example.

# Intel Tensorflow
- intel published `tensorflow` configuration that is 2X -4x times faster than out-of-the-box `tensorflow`

## Talks

### scikitlearn 0.21
- rf models are larger than gb trees
- histogram gradient boosted trees are implemented, low memory cost, faster training
- early stopping, stops training when a platuea of performance has been reached such as for number of trees.
- column transformer has been introduced, game changer, better alternative to `sklearn-pandas` 
- kbinsdicretizer preprocessor to compute nonelinear decision boundaries in order to generate new features which improves performance of linear models.

### Vaex
lazy loads large data from disk to RAM
creates virtual columns from column calculations
- has great histogram plotting features
- supports hdfs and appache arrow
- user defined functions, can be translated to C using numbajit
- window functions like aggregate in pandas
- comes with a lot of wrappers for python ml libraries, supports scikitlearn but does not support all scalers
- pca on data points on a map with x-y coordinates can be used to shift coordinate points representing a density grid tilted grid into a horizontally alligned grid 
- save all operations done in one df, can be saved as json and applied to a new dataframe. Replacement for pipelines


### Make Docker Images Safe

- Large docker images have many exposed vulnarbilities
- securety tools like `claire` can be used to scan them
- shells can be attached to the docker image
- use distroless image (reduced images)
- distroless get rid of everything that is not needed, for example the `ls` command from the shell.
- `pyinstaller` can be used to reduce dependencies, but does not detect all dependencies and then they need to be added manually.

Do:
- dont run as root¨
- use image hash instead of image name and tag (hash sign version of image similar like git commit hashes)
- build your own distroless images
- sign docker images


### Recommendation Engine

- use euclidean distance of ratings to calculate similarity score
- calculate weighted average (by similarity score) as prediction
- recommend if predicted recommendation is higher than average rating


### Python packaging

[slides](https://github.com/judy2k/publishing_python_packages_talk)

#### Dev Test workflow
- use `pip install -e .` install package from from current wd, during package dev. updates all files loaded from repo, like devtools::load_all() in R. Does not install directly into py dist.

put `.py` files into `src/` directory instead of packagename directory. Force your tests to run on `pip install -e .` version of your code.

Testing with `Tox` somehow like Rcmdcheck, different python versions can be spcified.

`check-manifest` checks whether all files are included in tar ball.

#### Documentation
sphinx pythonic solution for python doc. generates api reference from docstrings

mkdocs language agnostic markdown documentation for projects libraries, `pydocmd` claims to do generate api reference from docstrings. sphinx might be more advanced, on this, best to check receommended docstring layout recommended for the tool.

add readme.md to setuptools

```
from setuptools import setup
with open("README.md", "r") as fh:
 long_description = fh.read()
setup(
 …
 long_description=long_description,
 long_description_content_type="text/markdown", 
 ...
)
```

#### Templates

cookiecutter has python package templates

 