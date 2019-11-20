Model selection requires a good validation strategy that allows to test different models and their configurations and being able to realistically estimate the performance of the generated models for new data. In order to find the right balance between bias and variance, the dataset is traditionally split into training and validation set. Model training being done on the training set and performance validation is done on the validation set. In order to avoid parameter selection bias and additional test split can be created that can be used for final model evaluation as training and validation set are used for repeated trainings of different models and model configurations [9]. The smaller the data set, the more likely it is that creating random splits of the data will result in heterogeneous splits and thus introduce an additional source for bias. Therefore it is customary to repeat this procedure and creating multiple sets of splits, the gold standard being nested cross-validation [10]. Additionally, it is important that the data assigned to the different splits is truly independent and that no data is leaking from one split into the other. In our case we have several audits and inspections that have been performed on the same study and we need to make sure that data from one study can only be found in one split. Last but not least an adequate performance metric needs to be picked that considers class imbalance and model application requirements.

Most findings in our data set result from audits which are planned on an annual cycle strategy adjustments have been made each year that are not represented in our training data. Our data has been gathered over a stretch of 8 years where roughly 100 audits and inspections have taken place each year while only around 50 audits and inspections took place in 2018 due to an  IT systems migration. We are interested in modelling recurring patterns that allow us to identify risk areas for future audits. Therefore we reserved all data from 2017 and 2018 for testing excluding those studies which have already been audited before 2017. All data before 2017 was used for model tuning and repeatedly split into training and validation sets using 10-fold cross validation [10]. Our primary goal was to generate a model with well-calibrated probability predictions that reflects the actual risk of observing a specific finding. We therefore used the receiver operator control area under the curve (ROC-auc) as a performance metric to maximize during model parameter tuning because it evaluates the entire range of the classification predictions. Our secondary goal was to provide an interpretable model and therefore used the 1SE rule [11] when evaluating complexity against performance. 

Nevertheless the final test set was small and we were risking that this split introduces bias by chance. In order to account for this potential bias when calculating our final error estimate, we created annual rolling splits designating past data for 10-fold cross validation and next years future data (excluding studies that had already been audited) for testing. For each split we consecutively refitted our models using the best parameters as previously determined and calculated training and validation mean ROC-auc + SE (using k-fold cross validation stratified by study and audit/inspection) and test ROC-auc.


Handelma GS, Kok HK, Chandra RV, Razavi AH, Huang S, Brooks M, et al. Peering Into the Black Box of Artificial Intelligence: Evaluation Metrics of Machine Learning Methods. AJR Am J Roentgenol. 2018; 17:1–6

Varma S., Simon R. Bias in error estimation when using cross-validation for model selection, BMC Bioinformatics 2006, 7:91
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1397873/

[Breiman, Friedman, Stone, and Olshen's Classification and Regression Trees (1984). section 3.4.3.


### Purpose
- estimate performance
- model tuning/selection

#### Hazards/Risks 
- training bias (overfitting to training data)
- optimization bias (optimizing parameters to training data)
- selection bias (selection bias in training data)
- data leakage between splits
- Chance / Bad Luck
- not enough data

##### Split into Train, Valid
- eliminates training bias

##### Split into Train, Valid, Test
- eliminates optimization bias

#####  SME knowledge, Data Curation
- can reduce the risk of selection bias

#####  Stratified Sampling, careful feature selection
- ensures that there is no data leakage

#####  Multiple Data Splits (k-fold cv)
- reduces the risk of over/under estimating error by chance

#####  Learning Curve
- test whether model improves when getting more data
