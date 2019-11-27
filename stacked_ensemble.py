"""
This module defines a wrapper function to extract the information that are contained in HTML files automatically .
"""
import numpy as np
import random
import pandas as pd
import os, pickle
import csv
from itertools import combinations
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

__author__ = "Zheng Li"
__email__ = "zhengl@vt.edu"
__date__ = "Nov. 26, 2019"

class stacked_ensemble_regression():
    def __init__(self, sub_estimator, aggregator_estimator, feature_name, layers, model_number_layer, feature_ratio, sample_ratio,random_state):
        self.sub_estimator = {}
        for es in sub_estimator.keys():
            self.sub_estimator[es] = clone(sub_estimator[es])
        
        self.aggregator_estimator = {}
        for es in aggregator_estimator.keys():
            self.aggregator_estimator[es] = clone(aggregator_estimator[es])

        self.feature_name = feature_name
        self.columns = feature_name
        self.layers = layers
        self.model_number_layer = model_number_layer
        self.feature_ratio = feature_ratio
        self.sample_ratio = 1- sample_ratio
        self.random_state = random_state
        self.path = os.getcwd()

    def fit(self, X, Y):
        """
        Optimize the ensemble model parameters in the "sand box" layers with the training data.
        """
        # load the data in panda data framework
        X_df = pd.DataFrame(X, columns = self.feature_name)
        # delete the previous model parameters file and create a new one
        if os.path.exists(self.path + '/' + 'model_params'):
            os.system('rm -r model_params')
        os.mkdir('model_params')
        # enumerate and train all the sub-models in each layer 
        n = 0
        while n < self.layers:
            dir = self.path + '/model_params' + '/layer_' + str(n+1)
            os.mkdir(dir)
            num = self.model_number_layer[n]
            feature_gen = []
            feature_names = []
            DATA_params = {}
            for m in range(num):
                # select a random feature size according to the pre-defined ratio ("feature_ratio")
                columns_select = random.sample(self.feature_name, int(round(len(self.feature_name)* self.feature_ratio)))
                print ('columns_select', columns_select)
                if n == 0:
                    X_tr = X_df
                    X_tr = X_tr[columns_select].values.astype(np.float)
                else:
                    X_tr = self.X_tr[columns_select].values.astype(np.float)
                # feature preprocessing to standarize the feature for an improvement of training performance  
                scaler = preprocessing.StandardScaler()
                scaler.fit(X_tr)
                X_tr = scaler.transform(X_tr)
                # save the preprocessing parameters for prediction
                DATA_params[m] = {'mean' : scaler.mean_, 'variance': scaler.var_, 'columns': columns_select}
                # select a random sample size according to the pre-defined ratio ("sample_ratio") 
                X_train, X_test, Y_train, Y_test = train_test_split(X_tr, Y,\
                                                      test_size = self.sample_ratio, random_state= self.random_state)
                # optimize all the sub_model parameters 
                for es in self.sub_estimator:
                    self.sub_estimator[es].fit(X_train, Y_train.ravel())
                    joblib.dump(self.sub_estimator[es], dir+ '/' + es + '_' + str(m)+'.pkl')
                    feature = self.sub_estimator[es].predict(X_tr)
                    feature_gen.append(feature)
                    feature_names.append(es + '_' + str(m))
            # update 'X_tr' data for training the models at next layer  
            self.feature_name = feature_names
            self.X_tr = pd.DataFrame(np.array(feature_gen).T, columns = self.feature_name)
            # save the model parameters in pickle file
            output = open(dir+ '/' + 'params.pkl','wb')
            pickle.dump(DATA_params, output)
            output.close()
            n+=1
        # create folder for aggregation model
        self.dir_aggregator = self.path + '/model_params/' + 'aggregator'
        os.mkdir(self.dir_aggregator)
        # train the aggregator model using the data from the last layer
        for es in self.aggregator_estimator:
            self.aggregator_estimator[es].fit(self.X_tr, Y.ravel())
            joblib.dump(self.aggregator_estimator[es], self.dir_aggregator + '/' + es + '.pkl')

        return self
    
    def predict(self, X):
        """
        Ensemble model prediction using the trained model architectures. 
        """
        X_df = pd.DataFrame(X, columns = self.columns)
        if os.path.exists(self.path + '/' + 'model_params'):
            n = 0
            while n < self.layers:
                dir = self.path + '/model_params' + '/layer_' + str(n+1)
                # load in all the trained model parameters from pickle file
                f = open(dir + '/' + 'params.pkl', 'rb')
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                DATA_params = u.load()
                f.close()
                # load in the feature columns at each layer  
                feature_names = []
                feature_gen = []
                num = self.model_number_layer[n]
                for m in range(num):
                    columns_select = DATA_params[m]['columns']
                    print('columns_select', columns_select)
                    if n == 0:
                        X_ = X_df
                        X_ = X_[columns_select].values.astype(np.float)
                    else:
                        X_ = self.X_te[columns_select].values.astype(np.float)
                    # load in the standarization parameters
                    X_ = (X_ - DATA_params[m]['mean'])/np.sqrt(DATA_params[m]['variance'])
                    # model prediction on the new data
                    for es in self.sub_estimator:
                        sub_estimator = joblib.load(dir+ '/' + es+ '_' + str(m)+'.pkl')
                        feature = sub_estimator.predict(X_)
                        feature_gen.append(feature)
                        feature_names.append(es+ '_' + str(m))
                # update 'X_te' data for training the models at next layer  
                self.X_te = pd.DataFrame(np.array(feature_gen).T, columns = feature_names)
                n+=1
            # aggregator model prediction
            for es in self.aggregator_estimator:
                aggregator_model = joblib.load(self.dir_aggregator + '/' + es + '.pkl')
                prediction = aggregator_model.predict(self.X_te)
        else:
            raise ValueError('Invalid model parameter file or model_params file is missing')
     
        return prediction
