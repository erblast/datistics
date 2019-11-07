
# Basel Biometrics Society Seminar 20191101 Predictive modelling, machine learning and causality

## Clinical Predictions ML vs. Stats
Clinical Prediction Models
https://www.springer.com/gp/book/9780387772431

Log Regression vs ML
https://www.ncbi.nlm.nih.gov/pubmed/30763612

## The EQUATOR network and reporting guidelines for prediction models
Enhancing the Quality and Transperancy of healtth research

 Willi Sauerbrei
 https://www.equator-network.org/reporting-guidelines/
 
 Altman Moher Book
 https://onlinelibrary.wiley.com/doi/book/10.1002/9781118715598
 
 Why Most Published Research Findings Are False
 https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124
 
 
 ## Torsten Hothorn Score-based transformation learning
 
 http://user.math.uzh.ch/hothorn/
 
 **Boosting**  
 - fit baeline model
 - look at correlation of residuals with features
 - add feature with highest correlation to the model
 - repeat
 
 score-based transformation learning uses a score based on ecdf of the residuals
 `tram` R package http://ctm.r-forge.r-project.org/
 
 Transformation Trees and Forests `trtf`
 https://cran.r-project.org/web/packages/trtf/index.html
 http://ctm.r-forge.r-project.org/
 
 These models do not give a mean estimate as a prediction but a likelihood distribution

## Webpage with Conference Content Including Slide decks

https://baselbiometrics.github.io/20191101_ML/bbs1stNov2019.html
