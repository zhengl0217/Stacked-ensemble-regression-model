# Stacked ensemble regression python module. 

This repository contains the python source code of stacked ensemble model based on [scikit-learn](https://scikit-learn.org/stable/) library for regressional analysis, namely **stacked_ensemble.py**.
Motivated by the blog article of ["Why do stacked ensemble models win data science competitions?"](https://blogs.sas.com/content/subconsciousmusings/2017/05/18/stacked-ensemble-models-win-data-science-competitions/), I created this ensemble model to boost the predictive performance of conventional machine learning models. The methodology of the stacked ensemble model is very similar to the perceptron algorithm. During the training process, all the submodels randomly select a subset of features and samples to optimize the parameters and the output results of the previous layer become the input for training the submodels at the next layer. The below schematic shows the architecture of the stacked ensemble model. Even though some open-source platforms (e.g., [H2O](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html)) have provided the ensemble algorhtms with advanced functionality, they are not easy to learn and not flexible enough. I tested the model predictive performance using one of my previous first-principle datasets. The prediction performance is surprising well for the small test error compare to regular linear regression. However, this model has clear pros and cons as shown in the following:

Pros:
---
1. High prediction accuracy.
2. No need for feature selection.
3. Highly flexible model structure. 

Cons:
---
1. Model tends to be overfitted.
2. Training time is high.
3. Hyperparameter tuning is time consumption. 
4. Low interpretability.

![alt text](https://github.com/zhengl0217/Stacked-ensemble-regression-model/blob/master/model_schematics.png)
