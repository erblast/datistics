In this tutorial I want to show how you can use alluvial plots to
visualise model response in up to 4 dimensions. `easyalluvial` generates
artificial data space using fixed values for unplotted variables or uses
the partial dependence plotting method. It is model agnostic but offers
some convenient wrappers for `caret` models. <!--more-->

Introduction
============

Taking a peek
-------------

When building machine learning model we are usually faced with a
trade-off between accurracy and interpretability. However even if we
tend to lean towards accuracy and pick a modelling method that results
in nearly uninterpretable models we can still make use of a bunch of
model agnostic techniques that have been summarized in this excellent
ebook [Interpretable Machine Learning by Christoph
Molnar](https://christophm.github.io/interpretable-ml-book/).

Whithout getting to theoretical I personally always feel the urge to
simply take a peek simulate some data and see how the model reacts a
method described in [Wickham H, Cook D, Hofmann H (2015) Visualizing
statistical models: Removing the blindfold. Statistical Analysis and
Data Mining 8(4)
&lt;doi:10.1002/sam.11271&gt;](http://vita.had.co.nz/papers/model-vis.pdf).
In order to simulate data we can generate a a vector with a sequence of
values over the entire range of a predictor variable of interest while
setting all the others to their median or mode and use this artificial
data space to obtain model predictions which we can plot against our
variable of interest. An R package that will do this for you is
(`plotmo`)\[<https://cran.r-project.org/web/packages/plotmo/index.html>\].
Instead of ranging over 1 predictor variable we can create a data grid
using 2 predictor variables and plot the response as a third dimension.
However this is as far as you can go in a conventional plots. Alluvial
plots can line up much more than 3 dimensions on a plane next to each
other only limited by the number of flows as it will get too cluttered
when there are too many of them.

Which variables to plot
-----------------------

When using conventional model response plotting beeing limited two 2
variables we can simply resolve this by generating many plots and look
at them one by one. Alluvial plots require a bit more attention and
cannot easily be screened and compared since visually there is so much
going on. Therefore I do not recommend to brute force it by simply
creating a lot of random combinations of predictor variables and
multiple alluvial plots but instead to focus on those that have the
highest calculated feature importance. Feature importance values are
natively provided by most modelling packages. So the question is how
many can we plot and it turns out 4 features will usually result in well
interpretable plot.

Generating the data space
-------------------------

    suppressPackageStartupMessages( require(tidyverse) )
    suppressPackageStartupMessages( require(easyalluvial) )
    suppressPackageStartupMessages( require(mlbench) )
    suppressPackageStartupMessages( require(randomForest) )

We start by creating a model

    data('BostonHousing')
    df = as_tibble( BostonHousing )
    m = randomForest( lstat ~ ., df )

and looking at the importance features

    imp = m$importance %>%
      tidy_imp(df)

    knitr::kable(imp)

<table>
<thead>
<tr class="header">
<th align="left">vars</th>
<th align="right">imp</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">medv</td>
<td align="right">8352.4165</td>
</tr>
<tr class="even">
<td align="left">rm</td>
<td align="right">3370.7073</td>
</tr>
<tr class="odd">
<td align="left">indus</td>
<td align="right">2765.8359</td>
</tr>
<tr class="even">
<td align="left">age</td>
<td align="right">2716.4687</td>
</tr>
<tr class="odd">
<td align="left">crim</td>
<td align="right">2134.6400</td>
</tr>
<tr class="even">
<td align="left">dis</td>
<td align="right">1819.3780</td>
</tr>
<tr class="odd">
<td align="left">nox</td>
<td align="right">1696.9545</td>
</tr>
<tr class="even">
<td align="left">tax</td>
<td align="right">724.1518</td>
</tr>
<tr class="odd">
<td align="left">b</td>
<td align="right">682.6062</td>
</tr>
<tr class="even">
<td align="left">ptratio</td>
<td align="right">340.1351</td>
</tr>
<tr class="odd">
<td align="left">rad</td>
<td align="right">222.3688</td>
</tr>
<tr class="even">
<td align="left">chas</td>
<td align="right">131.2264</td>
</tr>
<tr class="odd">
<td align="left">zn</td>
<td align="right">101.5024</td>
</tr>
</tbody>
</table>

When generating the data space we cannot screent and infinite amount of
values per variable. We want to create all possible combinations between
the values of the 4 variables we want to plot and an alluvial plot we
cannot distinguish more than 1000 flows I recommend to go with 5 values
which will result in 5 x 5 X 5 X 5 --&gt; 625 combinations. That also
leaves some wiggeling room if one of the top 4 variables is a factor
with more than 5 levels. `get_data_space()` will split the range of a
variable into 3 and picks the median of each split and add the variable
minimum and the maximum to the set.

    dspace = get_data_space(df, imp
                            , degree = 4 # specifies the number of variables
                            , bins = 5 # the number of values per variable
                            )

    knitr::kable( head(dspace, 10) )

<table>
<thead>
<tr class="header">
<th align="right">medv</th>
<th align="right">rm</th>
<th align="right">indus</th>
<th align="right">age</th>
<th align="right">crim</th>
<th align="right">dis</th>
<th align="right">nox</th>
<th align="right">tax</th>
<th align="right">b</th>
<th align="right">ptratio</th>
<th align="right">rad</th>
<th align="left">chas</th>
<th align="right">zn</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">0.46</td>
<td align="right">2.90</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="even">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">0.46</td>
<td align="right">32.20</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="odd">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">0.46</td>
<td align="right">67.95</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="even">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">0.46</td>
<td align="right">94.60</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="odd">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">0.46</td>
<td align="right">100.00</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="even">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">4.27</td>
<td align="right">2.90</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="odd">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">4.27</td>
<td align="right">32.20</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="even">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">4.27</td>
<td align="right">67.95</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="odd">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">4.27</td>
<td align="right">94.60</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
<tr class="even">
<td align="right">5</td>
<td align="right">3.561</td>
<td align="right">4.27</td>
<td align="right">100.00</td>
<td align="right">0.25651</td>
<td align="right">3.20745</td>
<td align="right">0.538</td>
<td align="right">330</td>
<td align="right">391.44</td>
<td align="right">19.05</td>
<td align="right">5</td>
<td align="left">0</td>
<td align="right">0</td>
</tr>
</tbody>
</table>

Total rows in dspace: 625

    dspace %>%
      summarise_all( ~ length( unique(.) ) ) %>%
      knitr::kable()

<table>
<thead>
<tr class="header">
<th align="right">medv</th>
<th align="right">rm</th>
<th align="right">indus</th>
<th align="right">age</th>
<th align="right">crim</th>
<th align="right">dis</th>
<th align="right">nox</th>
<th align="right">tax</th>
<th align="right">b</th>
<th align="right">ptratio</th>
<th align="right">rad</th>
<th align="right">chas</th>
<th align="right">zn</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="right">5</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">1</td>
<td align="right">1</td>
<td align="right">1</td>
<td align="right">1</td>
<td align="right">1</td>
<td align="right">1</td>
<td align="right">1</td>
<td align="right">1</td>
<td align="right">1</td>
</tr>
</tbody>
</table>

Generating model response
-------------------------

    pred = predict(m, newdata = dspace)

Plotting
--------

    alluvial_model_response(pred, dspace, imp, degree = 4, bins = 5)

![](2019-03-29-visualising-model-response-with-easyalluvial_files/figure-markdown_strict/unnamed-chunk-7-1.png)

`easyalluvial` allows you to build exploratory alluvial plots (sankey
diagrams) with a single line of code while automatically binning
numerical variables. In version `0.2.0` marginal histograms improve the
visibility of the numerical variables. Further a method has been added
that creates model agnostic 4 dimensional partial dependence plots to
visualise model response.

Introduction
============

I am happy to announce the release of `easyalluvial 0.2.0` with some
exciting new features and some mayor changes compared to version `0.1.8`
Some improvements were made on the default plotting options which alter
the default plot output thus classify as mayor changes (see below).

    suppressPackageStartupMessages( require(tidyverse) )
    suppressPackageStartupMessages( require(easyalluvial) )

Marginal Histograms
===================

The automated binning of numerical variables hides their distribution.
This new feature alleviates that.

    p = alluvial_wide(mtcars2, max_variables = 9)

    p_grid = add_marginal_histograms(p, mtcars2)

{{&lt; wide-image src="/images/marg\_hits.png" title="\]" &gt;}}

4 Dimensional Partial Dependence Plots
======================================

Techniques for explaining the predictions of machine learning models
have been discussed a lot lately. For an introduction into the topic I
can recommend this excellent ebook [Interpretable Machine Learning by
Christoph Molnar](https://christophm.github.io/interpretable-ml-book/).
Inspired by packages like `plotmo` and `pdp` which can make partial
dependence plots (PDP) of 2 feature variables against the response
variable I tested whether alluvial plots, which can basically line up an
unlimited amount of features on a 2 dimensional plane, can be used to
make PDPs with more than 2 feature variables. And basically it turns out
that you can use 4 features before things get to cluttered. And this is
how it looks like:

    df = select(mtcars2, -ids)

    train = caret::train( disp ~ .
                          ,  df
                          , method = 'rf'
                          , trControl = caret::trainControl( method = 'none' )
                          , importance = TRUE )


    p = alluvial_model_response_caret(train, degree = 4, method = 'pdp')

    p_grid = add_marginal_histograms(p, df, plot = F) %>%
      add_imp_plot(p, df)

We see that the top 4 important features of the model have been selected
and that 5 values have been picked over the range of the numerical
variables which together with the levels of the categorical variable
have been used to construct a data grid of all possible combinations.
This data grid has been combined with the values of the remaining
features of each observation in the training data which then has been
used to generate model predictions which were consecutively averaged
([see the ebook for a better explanation of
PDPs](https://christophm.github.io/interpretable-ml-book/)).  
<br></br> {{&lt; wide-image src="/images/pdp.png" title="\]" &gt;}}

#### Step-by-Step

-   On the left we see the averaged model predictions that are generated
    by a specific combination of the 4 most important variables. You can
    find the combinations by tracing the coloured flows.
-   The stratum label of the individual feature variables indicate the
    variable value and which fraction of the colored flows pass through
    it.
-   On the right you see the feature importance of all variables and the
    proportion contributed by the plotted variables on the alluvial
    plot.
-   On the top left you see how the distribution of the generated
    predictions compare to the distribution of the predicted variable
    (in this case disp) in the training data.
-   The marginal histograms indicate the original distributions in the
    raining data and the lines indicate the location of the values
    picked for the data grid.

A more in-depth tutorial for this feature can be found on the [project's
github page](https://github.com/erblast/easyalluvial) which will also be
vailable on this blog in a few days.

If you are as enthusiastic about alluvial plots as me you will
appreciate this plot because it can help you to get an intuitive
mid-level understanding of how you're model is making the predictions.
Just be aware of a few limitations.

#### Limitations

-   There is a loss of information when binning the numerical variables
-   The combinations generated when making the grid might be outside the
    feature distribution space (generate combinations that are
    impossible)
-   We only look at the combination of 4 features and disregard the
    others

To alleviate this you can reduce the complexity of the model by reducing
features (take out correlating variables) or use additional model
exploration methods such as classical PDPs, ALE plots, Shapely values,
etc, ...

#### We do not have to use `caret`

*Note: importance is calculated differently when using this
implementation of random forest.*

    df = select(mtcars2, -ids)
    m = randomForest::randomForest( disp ~ ., df)
    imp = m$importance

    dspace = get_data_space(df, imp, degree = 4)

    pred = get_pdp_predictions(df, imp
                               , .f_predict = randomForest:::predict.randomForest
                               , m
                               , degree = 4
                               , bins = 5)


    p = alluvial_model_response(pred, dspace, imp, degree = 4, method = 'pdp')

    p_grid = add_marginal_histograms(p, df, plot = F) %>%
      add_imp_plot(p, df)

Changes in Default Plotting Settings
====================================

-   Default colors have been changed. The first 7 colors of
    `palette_qualitative()` the function that provides the default
    colors have hand-picked for better contrast.

-   The stratum fill color of the variable determining the flow fill
    color in `alluvial_wide()` hast been set to match the flow fill
    color.

-   label text size can now be modified via `stratum_label_size`
    parameter. Labels have gotten slightly bigger by default.

More changes
============

...
[NEWS.md](https://github.com/erblast/easyalluvial/blob/master/NEWS.md)
