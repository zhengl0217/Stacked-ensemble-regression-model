import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel as C, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from stacked_ensemble import stacked_ensemble_regression
import scipy
import matplotlib.pyplot as plt

# load in data
df = pd.read_csv('data.csv')
columns = df.columns.values.tolist()
#df = pd.DataFrame(data.values, columns = columns)

# outlier removal                                                                                                  
df = df[df[columns[1:]].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

# feature selection                                                                                                
features = df.columns.values.tolist()[2:]
inp = df[features].values.astype(np.float)
tar = df['CO']
print ('features', features)

# scaling input data                                                                                               
scaler = preprocessing.StandardScaler()                                                                    
scaler.fit(inp[:,0:])
inp = scaler.transform(inp[:,0:])

# ensemble model spesifications
RF = RandomForestRegressor(random_state=0)
KNN = GridSearchCV(neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform'), 
                   cv=5,param_grid={"n_neighbors": [x for x in range(1,21)]})#weights = ['uniform', 'distance']
LASSO = linear_model.LassoCV(cv=5, random_state=0)
KR = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
params = {'n_estimators': 350, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}#, 'min_samples_leaf': 5}, 'max_features': 30,   
GBM = ensemble.GradientBoostingRegressor(**params)

# aggregator model(Gaussian Process regression)
kernel = 1**2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1)
GP = GaussianProcessRegressor(alpha=1e-10, copy_X_train=True, kernel= kernel, n_restarts_optimizer=0, normalize_y=False, 
                              optimizer='fmin_l_bfgs_b', random_state=None)

# split datasets for training and prediction
X_train, X_test, Y_train, Y_test = train_test_split(inp, tar, test_size = 0.25, random_state= None)

# ensemble model training
sub_models = {'KNN': KNN, 'RF': RF, 'GP': GP, 'GBM': GBM, 'KR': KR}
aggregator_model = {'LASSO': LASSO} 

""" parameter expliation for the "stacked_ensemble_regression" model
sub_estimator (dict): sub-models dict file (e.g., {'model_name': model}) 
aggregator_estimator (dict): aggregator model dict file (e.g., {'model_name': model})
feature_name (list): list of feature name 
layers (int): number of layers 
model_number_layer (list): the number of each sub-model at each layer
feature_ratio (float): ratio of randomly selected features for training each model
sample_ratio (float): ratio of randomly selected samples size for training each model
"""
model = stacked_ensemble_regression(sub_estimator =sub_models, aggregator_estimator = aggregator_model, 
                                    feature_name = features, layers = 2, model_number_layer = [20, 10], 
                                    feature_ratio = 0.75, sample_ratio = 0.75,random_state = None)
model.fit(X_train, Y_train)

# ensemble model prediction
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Calculate Normal distribution plot
RMSE_tr = np.sqrt(np.mean((Y_train.ravel()- Y_train_pred.ravel())**2))
RMSE_te = np.sqrt(np.mean((Y_test.ravel()- Y_test_pred.ravel())**2))
print('RMSE_tr', RMSE_tr)
print('RMSE_te', RMSE_te)

# draw the parity plot for model performance evaluation
plt.plot(Y_train, Y_train_pred, 's', markerfacecolor= 'None', markersize=4.5, markeredgecolor='grey', markeredgewidth=1)
plt.plot(Y_test, Y_test_pred, 's', markerfacecolor= 'None', markersize=4.5, markeredgecolor='b', markeredgewidth=1)
plt.show()
