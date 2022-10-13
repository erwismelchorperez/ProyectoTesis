from audioop import cross
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from datetime import datetime
import numpy as np

import pandas as pd
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split
class XGBoost_:
    def __init__(self, dataset):
        self.param_grid = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
        self.dataset = dataset
        self.num_round = 2
        self.bts = None
        self.dtrain = xgb.DMatrix(self.dataset.get_xtrain(),self.dataset.get_ytrain())
        self.dvalidation = xgb.DMatrix(self.dataset.get_xvalidation(),self.dataset.get_yvalidation())
        self.f = open("logger/XGBoost"+str(datetime.now())+".txt",'w')

    def Regression(self):
        scores = cross_val_score(XGBRegressor(objective='reg:squarederror'), self.dataset.get_xtrain(), self.dataset.get_ytrain(), scoring='neg_mean_squared_error')
        print((-scores)**0.5)
        self.f.write("\nXGBoostRegression \n")
        self.f.write("scores:    " + str((-scores)**0.5))

    def Classification(self):
        scores = cross_val_score(XGBClassifier(), self.dataset.get_xtrain(), self.dataset.get_ytrain())
        print("Score Validación cruzada:     ",np.average(scores))
        self.f.write("\nXGBoostClassifier validation_cross\n")
        self.f.write("scores:   " + str(scores))
        self.f.write("Promeido de validación cruzada:   " + str(np.average(scores)))

        scores = cross_val_predict(XGBClassifier(), self.dataset.get_xtrain(), self.dataset.get_ytrain())
        self.f.write("\nXGBoostClassifier validation_predict\n")
        self.f.write("MatrixConfusion:   " + str(confusion_matrix(self.dataset.get_ytrain(), scores)))


    def ClassifierXGBClassifier(self):
        #xbg = xgb.XGBClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        #xbg = xgb.XGBClassifier()#mayor rendimiento
        xbg = xgb.XGBClassifier(n_estimators=100, learning_rate=0.02)
        xbg.fit(self.dataset.get_xtrain(),self.dataset.get_ytrain())
        predictions = xbg.predict(self.dataset.get_xtest())
        scores = accuracy_score(self.dataset.get_ytest(),predictions)
        print("accuracy:     ", scores)
        self.f.write("\nXGBoostClassifier Predict \n")
        self.f.write("scores:   " + str(scores))

    def CloseArchivo(self):
        self.f.close()

class XGBoost_Hyperparameters:
    def __init__(self):
        self.data = pd.read_csv('./dataset/wholesale.csv')
        self.X = self.data.drop('Channel',axis=1)
        self.Y = self.data['Channel']
        self.Convert_Labels()
        self.space = {'max_depth': hp.choice('max_depth', np.arange(2, 14, 1, dtype=int)),
            'gamma': hp.uniform('gamma', 1, 9),
            'reg_alpha' : hp.uniform('reg_alpha', 0.2, 0.8),
            'reg_lambda' : hp.uniform('reg_lambda', 0.2, 0.8),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight' : hp.quniform('min_child_weight', 2, 50, 2),
            'n_estimators': 180,
            'seed': 0
        }
        (self.x_train, self.x_test, self.y_train, self.y_test) = self.get_SplitDataset()
        self.objective()

        self.trails = Trials()
        print(self.space)
        best_hyperparams = fmin(fn = self.objective(),space = self.space,algo = tpe.suggest,max_evals = 500,trials = self.trails)
        print("The best hyperparameters are : ","\n")
        print(best_hyperparams)

    def Convert_Labels(self):
        self.Y[self.Y == 2] = 0
        self.Y[self.Y == 1] = 1

    def get_dataset(self):
        print("dataset:     ", self.data)

    def get_SplitDataset(self):
        (x_train, x_test, y_train, y_test) = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 0)
        return (x_train, x_test, y_train, y_test)

    def objective(self):
        print(self.space['n_estimators'], "    ",      (self.space['max_depth']))
        clf=xgb.XGBClassifier(
                        n_estimators = self.space['n_estimators'], max_depth = (self.space['max_depth']), gamma = self.space['gamma'],
                        reg_alpha = (self.space['reg_alpha']),min_child_weight=(self.space['min_child_weight']),
                        colsample_bytree=(self.space['colsample_bytree']))

        evaluation = [( self.x_train, self.y_train), ( self.x_test, self.y_test)]

        clf.fit(self.x_train, self.y_train,
                eval_set=evaluation, eval_metric="auc",
                early_stopping_rounds=10,verbose=False)


        pred = clf.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, pred>0.5)
        print ("SCORE:", accuracy)
        print('loss', -accuracy, 'status', STATUS_OK )

        return {'loss': -accuracy, 'status': STATUS_OK }
        #return {'loss': loss, 'params': params, 'status': STATUS_OK}
