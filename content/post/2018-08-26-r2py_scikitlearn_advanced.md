---
date : 2018-08-26
slug : r2py_scikitlearn_advanced
title : Moving from R to python - 6/8 - scikitlearn
author : Bjoern Koneswarakantha
categories: 
  - R vs. python
  - scikitlearn
  - sklearn-pandas
  - pipes
  - sparse data
tags: 
  - R vs. python
  - scikitlearn
  - randomized parameter search
  - sklearn-pandas
  - pipes
  - sparse data
summary : We look into some techniques for scikitlearn that allow us to write more                                        generalizable code that executes faster and helps us to avoid numpy arrays.
thumbnailImagePosition : left
thumbnailImage : r2py.png
---

{{< image classes="center" src="../../../r2py.png" thumbnail="../../../r2py.png" thumbnail-width="180px" thumbnail-height="180px">}}


- [1 of 8: IDE]( {{< relref "2018-08-21-r2py_ide.md" >}}  )
- [2 of 8: pandas]( {{< relref "2018-08-22-r2py_pandas.md" >}}  )
- [3 of 8: matplotlib and seaborn]( {{< relref "2018-08-23-r2py_matplotlib_seaborn.md" >}}  )
- [4 of 8: plotly]( {{< relref "2018-08-24-r2py_plotly.md" >}}  )
- [5 of 8: scikitlearn]( {{< relref "2018-08-25-r2py_scikitlearn.md" >}}  )
- [6 of 8: advanced scikitlearn]( {{< relref "2018-08-26-r2py_scikitlearn_advanced.md" >}}  )
- [7 of 8: automated machine learning]( {{< relref "2018-08-27-r2py_automated_ML.md" >}}  )


<!-- toc -->

# Advanced scikitlearn

In the last post, we have seen some advantages of `scikitlearn`. Most notably the seamless integration of parallel processing. I was struggeling a bit with the fact that `scikitlearn` only accepts `numpy` arrays as input and I was missing the `recipes` package which makes initial data transformation in `R` so much easier. Then I stumbled upon `sklearn-pandas` which seamlessly integrates `pandas` with `sklearn` without having to worry about numpy arrays and it supports a pipe based workflow, which is a `sklearn` feaure I have not started to explore yet. 

Apart from `sklearn-pandas` there are a number of projects that use the synthax and structure of scikit learn, a collection of them can be found at

-[http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html](https://github.com/scikit-learn-contrib/scikit-learn-contrib/blob/master/README.md)

-[http://scikit-learn.org/stable/related_projects.html](http://scikit-learn.org/stable/related_projects.html)





## `sklearn-pandas`

- [github](https://github.com/scikit-learn-contrib/sklearn-pandas)

Core of this package is the `DataFrameMapper` class which maps scikit learn Transformer classes to specific columns of a dataframe and outputs either a numpy array or dataframe.

Additionally it provides a `CategoricalImputer` which accepts categorical data, which I had to write myself before in the last post.


```python
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn

from sklearn_pandas import DataFrameMapper, CategoricalImputer, gen_features

df = sns.load_dataset('titanic')

X = df.copy().drop(['alive','survived'], axis = 'columns')
y = df.survived

# we need to set up transformations for numerical and categorical columns
col_categorical = list( X.select_dtypes(exclude=np.number) )
col_numerical   = list( X.select_dtypes(include=np.number) )

#we need to convert to list of lists
col_categorical = [ [x] for x in col_categorical ]
col_numerical   = [ [x] for x in col_numerical ]

# we have two ways of passing the classes as a simple list or as list of dicts if we need to pass
# arguments as well
classes_categorical = [ CategoricalImputer, sklearn.preprocessing.LabelBinarizer ]
classes_numerical = [ {'class':sklearn.preprocessing.Imputer, 'strategy' : 'median'}
                    , sklearn.preprocessing.StandardScaler
                    ]

# now that we have defined the columns and the classes of transformers we can use gen_features
# in order to generate a list of tuples suitable for DataFrameMapper

feature_def = gen_features(
    columns = col_categorical
    , classes = classes_categorical
)

feature_def_numerical = gen_features(
    columns = col_numerical
    , classes = classes_numerical
)

feature_def.extend(feature_def_numerical)

# when constructing the mapper we can specify whether we want a dataframe or a numpy array as output

mapper_df = DataFrameMapper( feature_def , df_out = True )

mapper_np = DataFrameMapper( feature_def , df_out = False )

mapped_df = mapper_df.fit_transform( df.copy() )

mapped_np = mapper_np.fit_transform( df.copy() )

print( mapped_np[1:10,1:20] )

mapped_df.head(10)
```

    [[1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0.]
     [0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 1. 0. 1. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 1. 0. 1. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1.]
     [0. 0. 1. 1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0.]]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>embarked_C</th>
      <th>embarked_Q</th>
      <th>embarked_S</th>
      <th>class_First</th>
      <th>class_Second</th>
      <th>class_Third</th>
      <th>who_child</th>
      <th>who_man</th>
      <th>who_woman</th>
      <th>...</th>
      <th>deck_G</th>
      <th>embark_town_Cherbourg</th>
      <th>embark_town_Queenstown</th>
      <th>embark_town_Southampton</th>
      <th>alone</th>
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.827377</td>
      <td>-0.565736</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>-0.502445</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.566107</td>
      <td>0.663861</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.786845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.827377</td>
      <td>-0.258337</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-1.566107</td>
      <td>0.433312</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.420730</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.827377</td>
      <td>0.433312</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.486337</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.827377</td>
      <td>-0.104637</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.478116</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.566107</td>
      <td>1.893459</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>0.395814</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.827377</td>
      <td>-2.102733</td>
      <td>2.247470</td>
      <td>0.767630</td>
      <td>-0.224083</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.827377</td>
      <td>-0.181487</td>
      <td>-0.474545</td>
      <td>2.008933</td>
      <td>-0.424256</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.369365</td>
      <td>-1.180535</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>-0.042956</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 27 columns</p>
</div>



the results are looking really good, its almost as good as `recipes`. However if we wanted to apply a boxcox transformation on top of it we would have to write our own `scikit-learn` like transformer. However the transformer will be added in a future version so I would not bother with that at the moment.

## To sparse or not to sparse

In the `python` data world data is considered to be *sparse* or *dense*. Which adresses the number of zeros in a matrix [wiki](https://en.wikipedia.org/wiki/Sparse_matrix). *sparse* means that you have a lot of them while *dense* means the opposite. There is no particular threshold but we should be aware that some data transformatios like dummy encoding make our data more *sparse*. A *sparse* matrix can be stored in a more memory efficient format such similar as a compressed image file and some algorithms can computationally leaverage this format to reduce computing time. [lasso](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_dense_vs_sparse_data.html) and [boosting gradient style algorithms](https://medium.com/sbc-group-blog/to-sparse-or-not-to-sparse-691483f87a53) seem to be able to profit from the sparse data format while others [neural nets, knn](https://medium.com/sbc-group-blog/to-sparse-or-not-to-sparse-691483f87a53) do not, and some like [randomForest](https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required) require the regular dense format and will otherwise raise an error. We can use `SciPy` to transform matrices to a dense format. We can measure the sparcity ratio as follows## Sparcity ratio

### Sparcity Ratio


```python
def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])
```

```python
print('sparcity ratio original data:', round( sparsity_ratio(X), 2) )
print('sparcity ratio tranformed data:', round( sparsity_ratio(mapped_np), 2) )
```

    sparcity ratio original data: 0.17
    sparcity ratio tranformed data: 0.56
    

The transformation have resulted in a matrix with a high sparcity thus we will test whether we might benefit from converting to a sparse matrix format


```python
from scipy import sparse
from time import time
from sklearn.tree import DecisionTreeClassifier


X_sparse = sparse.coo_matrix(mapped_np)

clf_sparse = DecisionTreeClassifier()
clf_dense = DecisionTreeClassifier()

t0 = time()
clf_sparse.fit(X_sparse, y)
print('exec time sparse:', round( time() - t0,3 ) )


t0 = time()
clf_dense.fit(mapped_np, y)
print('exec time dense :', round( time() - t0,3 ) )
```

    exec time sparse: 0.019
    exec time dense : 0.008
    

We can see that our decision tree classifiert does not benefit from  a sparse data format.

## Pipelines

Pipelines are constructs that chain scikit preprocessing steps together and attaching an optional classifier or a regressor to the end. 

We can then use the pipe as we would use a regular model we can fit it and get predictions, we could get crossvalidated performance scores or perform parameter tuning. This has a couple of advantages.

- The code becomes more compact and readable
- Instead of saving multiple transformers (scaling, boxcox ) we can simply store one to apply to future data
- We can tune several steps of the pipeline in one go (for example feature selector + model tuning parameters)

We are going to contruct two pipes one for preprocessing and one for model fitting. It makes sense to seperate these two because we the first one contains a defined sequence of steps and the last pipe we are going to use to tune certain parameters via cross validation. 

When performning the cross validation the transformers and estimators in the pipe will be applied **after** splitting the data into cross validation pairs. Cross validation is computationally expensive and we only want to use it for steps which are likely to introduce bias and can lead to overfitting such as feature selection and hyperparameter tuning.


### Preprocessing Pipeline

We are going to apply the `sklearn-pandas` dataframe mapper and a low variance feature filter.


```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import os


pipe_pre_process = sklearn.pipeline.Pipeline([
    ('mapper', mapper_np ) 
    , ('feats', VarianceThreshold() )
])


pipe_pre_process
```

    Pipeline(memory=None,
         steps=[('mapper', DataFrameMapper(default=False, df_out=False,
            features=[(['sex'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]), (['embarked'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_la...h_std=True)])],
            input_df=False, sparse=False)), ('feats', VarianceThreshold(threshold=0.0))])




```python
pipe_pre_process.named_steps
```

    {'feats': VarianceThreshold(threshold=0.0),
     'mapper': DataFrameMapper(default=False, df_out=False,
             features=[(['sex'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]), (['embarked'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]), (['class'], [CategoricalImp...es='NaN', strategy='median', verbose=0), StandardScaler(copy=True, with_mean=True, with_std=True)])],
             input_df=False, sparse=False)}



The parameters are saved as follows in a nested dictionary and are named after the following principle `step_name + '__' + argument`


```python
pipe_pre_process.get_params()
```

    {'feats': VarianceThreshold(threshold=0.0),
     'feats__threshold': 0.0,
     'mapper': DataFrameMapper(default=False, df_out=False,
             features=[(['sex'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]), (['embarked'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]), (['class'], [CategoricalImp...es='NaN', strategy='median', verbose=0), StandardScaler(copy=True, with_mean=True, with_std=True)])],
             input_df=False, sparse=False),
     'mapper__default': False,
     'mapper__df_out': False,
     'mapper__features': [(['sex'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['embarked'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['class'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['who'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['adult_male'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['deck'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['embark_town'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['alone'],
       [CategoricalImputer(copy=True, missing_values='NaN'),
        LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]),
      (['pclass'],
       [Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0),
        StandardScaler(copy=True, with_mean=True, with_std=True)]),
      (['age'],
       [Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0),
        StandardScaler(copy=True, with_mean=True, with_std=True)]),
      (['sibsp'],
       [Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0),
        StandardScaler(copy=True, with_mean=True, with_std=True)]),
      (['parch'],
       [Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0),
        StandardScaler(copy=True, with_mean=True, with_std=True)]),
      (['fare'],
       [Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0),
        StandardScaler(copy=True, with_mean=True, with_std=True)])],
     'mapper__input_df': False,
     'mapper__sparse': False,
     'memory': None,
     'steps': [('mapper', DataFrameMapper(default=False, df_out=False,
               features=[(['sex'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]), (['embarked'], [CategoricalImputer(copy=True, missing_values='NaN'), LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)]), (['class'], [CategoricalImp...es='NaN', strategy='median', verbose=0), StandardScaler(copy=True, with_mean=True, with_std=True)])],
               input_df=False, sparse=False)),
      ('feats', VarianceThreshold(threshold=0.0))]}



We can set a parameter


```python
pipe_pre_process.set_params(feats__threshold = 0.05)
pipe_pre_process.named_steps.feats
```

    VarianceThreshold(threshold=0.05)



Then we fit the preprocessing pipe to the data


```python
pipe_pre_process.fit(X)
X_proc = pipe_pre_process.fit_transform(X)

X_proc
```

    array([[ 1.        ,  0.        ,  0.        , ...,  0.43279337,
            -0.47367361, -0.50244517],
           [ 0.        ,  1.        ,  0.        , ...,  0.43279337,
            -0.47367361,  0.78684529],
           [ 0.        ,  0.        ,  0.        , ..., -0.4745452 ,
            -0.47367361, -0.48885426],
           ...,
           [ 0.        ,  0.        ,  0.        , ...,  0.43279337,
             2.00893337, -0.17626324],
           [ 1.        ,  1.        ,  0.        , ..., -0.4745452 ,
            -0.47367361, -0.04438104],
           [ 1.        ,  0.        ,  1.        , ..., -0.4745452 ,
            -0.47367361, -0.49237783]])



We will be using the pre processed data in another post so we are saving it to disc. We are storing it in feather format which is basically hdfs which has much faster in terms of reading and writing from and to disc.


```python
import feather
import os

if not os.path.isdir('./data'):
    os.mkdir('./data')


df_feather = mapped_df.\
    assign( y = y )

feather.write_dataframe(df_feather, './data/mapped_df.feather')
```

### Modelling Pipeline

We will add a feature selection step, which choses variables based on a univariate test such as a chisquare test (which we cannot use here because it does not accept negative values) and ANOVA and then fit a decision tree.


```python
pipe_mod = sklearn.pipeline.Pipeline([
    ('feats', sklearn.feature_selection.SelectKBest( k = 10) ) 
    , ('tree', sklearn.tree.DecisionTreeClassifier() )
])
```

We can apply the same '__' synthax as we used for setting the parameters of the pipe for constructing the dictionary for the sandomized hyperparameter search


```python
param_dist = dict( tree__min_samples_split = stats.randint(2,250)
                 , tree__min_samples_leaf = stats.randint(1,500)
                 , tree__min_impurity_decrease = stats.uniform(0,1)
                 , tree__max_features = stats.uniform(0,1)
                 , feats__score_func = [sklearn.feature_selection.f_classif ## Anova
                                       , sklearn.feature_selection.mutual_info_classif] ) ## nearest n

n_iter = 500

random_search = RandomizedSearchCV(pipe_mod
                                   , param_dist
                                   , n_iter = n_iter
                                   , scoring = 'roc_auc'
                                   , cv = RepeatedKFold( n_splits = 5, n_repeats = 3 )
                                   , verbose = True
                                   , n_jobs = 4 ## parallel processing
                                   , return_train_score = True
                                  )


random_search.fit(X = X_proc, y =  df.survived )
```

    Fitting 15 folds for each of 500 candidates, totalling 7500 fits
    

    [Parallel(n_jobs=4)]: Done  49 tasks      | elapsed:    5.8s
    [Parallel(n_jobs=4)]: Done 740 tasks      | elapsed:   34.9s
    [Parallel(n_jobs=4)]: Done 2005 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=4)]: Done 3737 tasks      | elapsed:  2.6min
    [Parallel(n_jobs=4)]: Done 5556 tasks      | elapsed:  4.3min
    [Parallel(n_jobs=4)]: Done 7241 tasks      | elapsed:  5.9min
    [Parallel(n_jobs=4)]: Done 7493 out of 7500 | elapsed:  6.1min remaining:    0.2s
    [Parallel(n_jobs=4)]: Done 7500 out of 7500 | elapsed:  6.1min finished
    




    RandomizedSearchCV(cv=<sklearn.model_selection._split.RepeatedKFold object at 0x000001916A6DC9B0>,
              error_score='raise',
              estimator=Pipeline(memory=None,
         steps=[('feats', SelectKBest(k=10, score_func=<function f_classif at 0x000001916A6ABF28>)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))]),
              fit_params=None, iid=True, n_iter=500, n_jobs=4,
              param_distributions={'tree__min_impurity_decrease': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001916A6DC160>, 'feats__score_func': [<function f_classif at 0x000001916A6ABF28>, <function mutual_info_classif at 0x000001916A6CD2F0>], 'tree__max_features': <scipy.stats._distn_infrastru...tree__min_samples_leaf': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001916A6D9FD0>},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score=True, scoring='roc_auc', verbose=True)




```python
random_search.best_estimator_
```

    Pipeline(memory=None,
         steps=[('feats', SelectKBest(k=10,
          score_func=<function mutual_info_classif at 0x000001916A6CD2F0>)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=0.6995334988533182, max_leaf_nodes=None,
                min_impurity_decrease=0.00253...t=47, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best'))])




```python
res = pd.DataFrame( random_search.cv_results_ )

res.sort_values('rank_test_score').head(10)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>mean_score_time</th>
      <th>mean_test_score</th>
      <th>mean_train_score</th>
      <th>param_feats__score_func</th>
      <th>param_tree__max_features</th>
      <th>param_tree__min_impurity_decrease</th>
      <th>param_tree__min_samples_leaf</th>
      <th>param_tree__min_samples_split</th>
      <th>params</th>
      <th>...</th>
      <th>split7_test_score</th>
      <th>split7_train_score</th>
      <th>split8_test_score</th>
      <th>split8_train_score</th>
      <th>split9_test_score</th>
      <th>split9_train_score</th>
      <th>std_fit_time</th>
      <th>std_score_time</th>
      <th>std_test_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>0.360021</td>
      <td>0.001642</td>
      <td>0.851727</td>
      <td>0.862858</td>
      <td>&lt;function mutual_info_classif at 0x000001916A6...</td>
      <td>0.699533</td>
      <td>0.00253909</td>
      <td>33</td>
      <td>47</td>
      <td>{'feats__score_func': &lt;function mutual_info_cl...</td>
      <td>...</td>
      <td>0.772863</td>
      <td>0.866083</td>
      <td>0.847184</td>
      <td>0.881330</td>
      <td>0.867859</td>
      <td>0.867060</td>
      <td>0.031329</td>
      <td>0.003972</td>
      <td>0.036028</td>
      <td>0.016553</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.004166</td>
      <td>0.001308</td>
      <td>0.829953</td>
      <td>0.838292</td>
      <td>&lt;function f_classif at 0x000001916A6ABF28&gt;</td>
      <td>0.87121</td>
      <td>0.0133541</td>
      <td>89</td>
      <td>210</td>
      <td>{'feats__score_func': &lt;function f_classif at 0...</td>
      <td>...</td>
      <td>0.736742</td>
      <td>0.809940</td>
      <td>0.845372</td>
      <td>0.858085</td>
      <td>0.798939</td>
      <td>0.825771</td>
      <td>0.006909</td>
      <td>0.003953</td>
      <td>0.034927</td>
      <td>0.019111</td>
    </tr>
    <tr>
      <th>194</th>
      <td>0.298661</td>
      <td>0.000000</td>
      <td>0.811322</td>
      <td>0.811516</td>
      <td>&lt;function mutual_info_classif at 0x000001916A6...</td>
      <td>0.813777</td>
      <td>0.0138638</td>
      <td>124</td>
      <td>79</td>
      <td>{'feats__score_func': &lt;function mutual_info_cl...</td>
      <td>...</td>
      <td>0.713745</td>
      <td>0.779844</td>
      <td>0.815275</td>
      <td>0.821707</td>
      <td>0.798939</td>
      <td>0.825771</td>
      <td>0.029102</td>
      <td>0.000000</td>
      <td>0.032071</td>
      <td>0.018513</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0.435979</td>
      <td>0.003441</td>
      <td>0.777916</td>
      <td>0.776460</td>
      <td>&lt;function mutual_info_classif at 0x000001916A6...</td>
      <td>0.407913</td>
      <td>0.127602</td>
      <td>3</td>
      <td>215</td>
      <td>{'feats__score_func': &lt;function mutual_info_cl...</td>
      <td>...</td>
      <td>0.754735</td>
      <td>0.786423</td>
      <td>0.768997</td>
      <td>0.766081</td>
      <td>0.750393</td>
      <td>0.787890</td>
      <td>0.268832</td>
      <td>0.005945</td>
      <td>0.019844</td>
      <td>0.009008</td>
    </tr>
    <tr>
      <th>285</th>
      <td>0.312650</td>
      <td>0.001333</td>
      <td>0.777642</td>
      <td>0.779836</td>
      <td>&lt;function mutual_info_classif at 0x000001916A6...</td>
      <td>0.95935</td>
      <td>0.0960747</td>
      <td>234</td>
      <td>36</td>
      <td>{'feats__score_func': &lt;function mutual_info_cl...</td>
      <td>...</td>
      <td>0.713745</td>
      <td>0.779844</td>
      <td>0.779288</td>
      <td>0.780731</td>
      <td>0.750393</td>
      <td>0.787890</td>
      <td>0.020375</td>
      <td>0.001885</td>
      <td>0.025972</td>
      <td>0.005002</td>
    </tr>
    <tr>
      <th>347</th>
      <td>0.003999</td>
      <td>0.001600</td>
      <td>0.777642</td>
      <td>0.779836</td>
      <td>&lt;function f_classif at 0x000001916A6ABF28&gt;</td>
      <td>0.722103</td>
      <td>0.0547579</td>
      <td>104</td>
      <td>224</td>
      <td>{'feats__score_func': &lt;function f_classif at 0...</td>
      <td>...</td>
      <td>0.713745</td>
      <td>0.779844</td>
      <td>0.779288</td>
      <td>0.780731</td>
      <td>0.750393</td>
      <td>0.787890</td>
      <td>0.002529</td>
      <td>0.001959</td>
      <td>0.025972</td>
      <td>0.005002</td>
    </tr>
    <tr>
      <th>263</th>
      <td>0.004570</td>
      <td>0.001416</td>
      <td>0.777642</td>
      <td>0.779836</td>
      <td>&lt;function f_classif at 0x000001916A6ABF28&gt;</td>
      <td>0.83097</td>
      <td>0.0618748</td>
      <td>211</td>
      <td>14</td>
      <td>{'feats__score_func': &lt;function f_classif at 0...</td>
      <td>...</td>
      <td>0.713745</td>
      <td>0.779844</td>
      <td>0.779288</td>
      <td>0.780731</td>
      <td>0.750393</td>
      <td>0.787890</td>
      <td>0.006060</td>
      <td>0.002616</td>
      <td>0.025972</td>
      <td>0.005002</td>
    </tr>
    <tr>
      <th>459</th>
      <td>0.005208</td>
      <td>0.000000</td>
      <td>0.777642</td>
      <td>0.779836</td>
      <td>&lt;function f_classif at 0x000001916A6ABF28&gt;</td>
      <td>0.93767</td>
      <td>0.0945291</td>
      <td>11</td>
      <td>96</td>
      <td>{'feats__score_func': &lt;function f_classif at 0...</td>
      <td>...</td>
      <td>0.713745</td>
      <td>0.779844</td>
      <td>0.779288</td>
      <td>0.780731</td>
      <td>0.750393</td>
      <td>0.787890</td>
      <td>0.009316</td>
      <td>0.000000</td>
      <td>0.025972</td>
      <td>0.005002</td>
    </tr>
    <tr>
      <th>224</th>
      <td>0.002137</td>
      <td>0.001575</td>
      <td>0.773051</td>
      <td>0.774232</td>
      <td>&lt;function f_classif at 0x000001916A6ABF28&gt;</td>
      <td>0.509988</td>
      <td>0.0569047</td>
      <td>74</td>
      <td>12</td>
      <td>{'feats__score_func': &lt;function f_classif at 0...</td>
      <td>...</td>
      <td>0.754735</td>
      <td>0.786423</td>
      <td>0.779288</td>
      <td>0.780731</td>
      <td>0.750393</td>
      <td>0.787890</td>
      <td>0.001999</td>
      <td>0.003991</td>
      <td>0.021037</td>
      <td>0.015873</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.004166</td>
      <td>0.002083</td>
      <td>0.771509</td>
      <td>0.774650</td>
      <td>&lt;function f_classif at 0x000001916A6ABF28&gt;</td>
      <td>0.555193</td>
      <td>0.0792222</td>
      <td>173</td>
      <td>195</td>
      <td>{'feats__score_func': &lt;function f_classif at 0...</td>
      <td>...</td>
      <td>0.713745</td>
      <td>0.779844</td>
      <td>0.779288</td>
      <td>0.780731</td>
      <td>0.750393</td>
      <td>0.787890</td>
      <td>0.006909</td>
      <td>0.005311</td>
      <td>0.027898</td>
      <td>0.013231</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 45 columns</p>
</div>



## Summary

We can see that our maximum ROC score 0.86 is similar to what we obtained in the last post (0.85) where we took a more manual approach. However using `sklearn-pandas` and pipes we were able to write code that is more generalizable and is less dependent on the actual dataset. We have more or less written a generalizable code for the preprocessing pipe however the code for the modelling pipe is quite specific for the model that we used. I f we wanted to train more models of a different type we would have to manually write pipes for them as well. 
