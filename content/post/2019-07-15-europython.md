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
summary : Europython 2019 Conference Notes
thumbnailImagePosition : left
thumbnailImage : https://pbs.twimg.com/profile_images/1082571058832703488/1QVxP99r.jpg
---


<!-- toc -->

# Europython 2019

I attended [europython 2019](https://ep2019.europython.eu/) here are some of my takeaways and notes.

## General Takeaways
- `docker` is widely used for reproducible environments especially for training sessions and for deploying models.
- mostly `docker` images would be deployed as `flask` apps, with an REST API.
- REST APIs can be managed using `swagger`
- `docker` images would be managed using `kubernetes`
- `continuous delivery` was advertised a lot
- `VS Code` seems to be the most popular IDE, very good demo at the MS booth. Beats `atom` and `pycharm` in my opinion.
- `scikitlearn 0.21` release great new features ColumnTransformer replaces `sklearn-pandas`, histogram gradient boosting, faster and more light weight then regular gradient boosting.
- `dirty_cat` has great encoders for dirty data. SimilarityEncoder and

## Training Days

### REST API/Microservices
- use `connexion` with `swagger` to build YAML configurable REST API's. `swagger` provides documentation and user interface based on yaml.

### Pytest
- pytest can run tests from other testing suites, in order to incorporate examples from docstrings doctest can be used as `pytest --doctest`
- pytest goes by no API is a good API
- markers can be used to organise tests
- fixtures can be used to pass data to tests

- `hypothesis` can be used to generate random testing strings. If a fail is  detected it will give a minimal reproducible example.

### Intel Tensorflow
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
- dont run as root
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
- use `pip install -e .` install package from from curent wd, during package dev. updates all files loaded from repo, like devtools::load_all() in R. Does not install directly into py dist.

put `.py` files into `src/` directory instead of packagename directory. Force your tests to run on `pip install -e .` version of your code.

Testing with `Tox` somehow like Rcmdcheck, different python versions can be spcified.

`check-manifest` checks whether all files are included in tar ball.

#### Documentation
sphinx pythonic solution for python doc. generates api reference from docstrings

mkdocs language agnostic markdown documentation for projects libraries, `pydocmd` claims to do generate api reference from docstrings. sphinx might be more advanced, on this, best to check receommended docstring layout recommended for the tool.

`pydoc` can also bve used to make documentation from docstrings. In standard library.

add readme.md to setuptools so it propably shows up on python

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

`cookiecutter` has python package templates

### Dirty Data

#### Dirty categorical features
  - manualy break up into two or more seperate features, for example first name, last name
  - manualy group categories
  - Similarity Endoding, similarity distance to category, new strings can be fitted on old categories
  - Jaro-winkler, levenstain, 3-gram similarity scores
  - `dirty_cat` has similarity encoder `from dirty_cat import SimilarityEncoder`
  - TargetEncoder, Encode Categorical Feature as Mean/Median of other value, example police officer ranking on Salary
  - Latent Category Encoder, builds new categories based on substring similarities

#### Missing Values
  - classical data generation assumption, data generation is complete and random entries are random.
  - NA values are seldom random, and sometimes are the result of the data model, like age of spouse will be NA for people that are single
  - mean imputation distorts the distribution, concerning for statisitcal models but not algorithmic models
  - when imputing age of spouse, missingness indicator could be used to flagg single people


### From Script to Open Source

-`docopt` helps you build GNU compliant CLI-tools
- code guides, only 2 parameters per function. 
- `python setup.py develop` same as `pip install -e .`
- setup.py let's you define entry points (for package plug-ins) and CLI callable name
- requirements.txt file can augment setup.py dependencies, stating tested dependency versions
- `black` reformats code to be pythonic
- `pre-commit` runs formatters such as `black` before git commit
- `flake8` to check you code
- `tox.ini` configuration file for `black` code standard and `flake8`
- use static type analysis, `MyPy` checks if function with wrong type has been called
- `tox` manages all those tools including testing tools similar to Rcmdcheck I guess
- `travisCI` pip installing `tox` is enough to run all test
- requirements updater is a bot that will continuosly check versions of dependencies, `PyUP`
- `pytest-cov` will check test coverage
- automated code review `PR`
- automated pull request merge `mergify`
- `twine` to upload to pypi

[blog](http://michal.karzynski.pl)

### State of Production ML in 2019

[slides + example projects](https://github.com/EthicalML/state-of-mlops-2019)

#### GITOPS STRATEGIES FOR ML
CI/CD via github, using, docker, kubernetes
[description](https://dzone.com/articles/a-practical-guide-to-operating-kubernetes-the-gito)

#### Modelling Process
- data assessment
- model assessment (feature importance, shap-values, pdp-plots, interpretability)
- production monitoring (see that asassments remain intact during production)
- explainer, model that adds explanations to predictions,
  * `alibi`, delivers pertinent negative and pertinate positive (minimum changes for positive and negative prediction)
- `seldon` can be used to manage kubernetes

#### Reproducibility
- Container Versioning

### Modern Continuous Delivery
[slides](tinyurl.com/moderncd)  
- deploy to production from commit #1  
- take over release schedule from IT to Business  
- CDEV is concept, CI and CDEP are techniques  
- Modern  
  * immutable infrastructure  
  * container orchestration  
  * version control and automation  
  * cloud native apps  

tools? choice or lock-in?, lock-in choices should be avoided

- `cookiecutter` seems to be what devtools/usethis is for python, can be used to setup CDEV for projects.
- generate + seal your secrets, otherwise you cant continuously deliver
- dont overload your yaml
- test-driven, pair programming
- the only way to go fast is to go well, robert c. martin


### Practical Clean Architecture

#### Typing for data interfaces
- typing, use type annotations when writing functions
- typing package has objects that allow you type specifications for dictionaries
- python 3.7 offers data classes that make this easier
- dataclasses can be frozen, immutability can be added
- use abstractions to interact with databases, `ABC` packages
- we can use `injector` to build interfaces, will inject stuff into data classes
- the interfaces can be documented with swagger
- make an in RAM db for testing

These interfaces are easily testable

#### Architecture
- lifetime 10 years
- make an application centric infrastructure
- do not put your db the center of your architecture
